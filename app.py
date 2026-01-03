# streamlit run app.py
import streamlit as st
import pandas as pd
import calendar
from datetime import datetime
from scheduler import ServiceScheduler

# -----------------------------------------------------------------------------
# 스타일 함수
# -----------------------------------------------------------------------------
def color_schedule_cells(val):
    color = 'white'; font_weight = 'normal'
    str_val = str(val)
    if str_val == '주': color = '#CDECFF'; font_weight = 'bold'
    elif str_val == '야': color = '#FFFF66'; font_weight = 'bold'
    elif str_val == '비': color = '#F5F5F5'
    elif str_val == '휴': color = '#FFEBEE'
    elif str_val == '교': color = '#E8F5E9'
    
    if isinstance(val, (int, float)) or '인원' in str(val):
        font_weight = 'bold'; color = '#EEEEEE'
    return f'background-color: {color}; color: black; font-weight: {font_weight}'

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="사회복무요원 근무표 생성기", layout="wide")
if 'step' not in st.session_state: st.session_state.step = 1
if 'config' not in st.session_state: st.session_state.config = {}

def step1_setup():
    st.title("Step 1: 기본 설정")
    st.info("근무표를 생성할 연도와 월, 그리고 요원들의 이름을 입력해주세요.")
    with st.form("setup_form"):
        col1, col2 = st.columns(2)
        now = datetime.now()
        year = col1.number_input("연도", 2023, 2030, now.year)
        month = col2.number_input("월", 1, 12, now.month)
        agents_input = st.text_area("요원 이름 (콤마 구분)", value="정다운, 김민성, 성민용, 한동현, 진유진, 이도원, 손창우")
        
        if st.form_submit_button("다음 단계로"):
            agent_list = [name.strip() for name in agents_input.split(',') if name.strip()]
            if not agent_list:
                st.error("요원 이름을 입력하세요.")
            else:
                st.session_state.config = {'year': year, 'month': month, 'agents': agent_list}
                st.session_state.step = 2
                st.rerun()

# -----------------------------------------------------------------------------
# Step 2
# -----------------------------------------------------------------------------
def step2_constraints():
    st.title("Step 2: 근무 설정 및 생성")
    
    cfg = st.session_state.config
    year = int(cfg['year'])
    month = int(cfg['month'])
    agents = cfg['agents']
    _, last_day = calendar.monthrange(year, month)
    
    korean_days = ["월", "화", "수", "목", "금", "토", "일"]
    date_columns = []
    for day in range(1, last_day + 1):
        weekday_index = calendar.weekday(year, month, day)
        date_columns.append(f"{day}({korean_days[weekday_index]})")

    # 사이드바 정보
    st.sidebar.header("ℹ️ 알고리즘 정보")
    st.sidebar.info("""
    **전역자 처리:**
    전역일 이후는 '교(5)'로 자동 채움 되며,
    전역일 이후의 휴무만큼 목표 휴무일이 차감됩니다.
    
    **자동 배정:**
    1. 선호도(주간/야간) 반영
    2. 목표 휴무일수 준수
    3. 최소 인원 및 최대 인원 제한 준수
    """)
    
    st.markdown(f"### 📅 {year}년 {month}월 설정")

    # --- 1. 요원별 특수 사항 (선호도 + 전역일) ---
    st.subheader("1. 요원별 특수 사항")
    
    preferences = {}
    discharge_dates = {}
    
    # 보기 좋게 3열로 배치
    cols = st.columns(3)
    for i, agent in enumerate(agents):
        with cols[i % 3]:
            # 카드처럼 보이게 컨테이너 사용
            with st.container(border=True):
                st.markdown(f"**👤 {agent}**")
                
                # 1) 선호 근무
                pref = st.selectbox(
                    "선호 근무", 
                    options=["선호 없음", "주간 선호", "야간 선호"],
                    key=f"pref_{agent}",
                    label_visibility="collapsed" # 공간 절약
                )
                preferences[agent] = pref
                
                # 2) 전역일 (0이면 전역 아님)
                d_date = st.number_input(
                    "전역일 (없으면 0)",
                    min_value=0, max_value=last_day, value=0,
                    key=f"disch_{agent}",
                    help=f"{agent} 요원이 이 날짜까지만 근무합니다."
                )
                discharge_dates[agent] = d_date

    st.markdown("---")
    st.subheader("2. 고정 근무 지정 & 생성")

    # 그리드 초기화
    if 'schedule_df' not in st.session_state:
        st.session_state.schedule_df = pd.DataFrame("", index=agents, columns=date_columns)
    else:
        if st.session_state.schedule_df.shape[1] != len(date_columns):
             st.session_state.schedule_df = pd.DataFrame("", index=agents, columns=date_columns)
        else:
             st.session_state.schedule_df.columns = date_columns
             st.session_state.schedule_df = st.session_state.schedule_df.astype(str).replace('nan', '')

    st.info("**입력 가이드:** 0:주 | 1:야 | 2:비 | 3:휴 | 4:불가 | 5:교육 (빈칸: 자동)")
    
    column_config_settings = {
        col: st.column_config.TextColumn(col, width="small", validate="^[0-5]?$") 
        for col in date_columns
    }

    edited_df = st.data_editor(
        st.session_state.schedule_df,
        use_container_width=True,
        column_config=column_config_settings
    )

    st.markdown("---")
    
    c1, c2 = st.columns([1, 5])
    if c1.button("⬅️ 뒤로"):
        st.session_state.step = 1
        st.rerun()
        
    if c2.button("🚀 알고리즘 구동", type="primary"):
        with st.spinner("특수 사항을 반영하여 최적의 근무표를 생성 중입니다..."):
            # 수정된 스케줄러 호출 (preferences, discharge_dates 전달)
            scheduler = ServiceScheduler(edited_df, year, month, preferences, discharge_dates)
            success, result_df, msg = scheduler.run()
            
            if success:
                st.success(f"✅ {msg}")
                
                # 1. 통계 테이블
                stats_data = []
                for agent in result_df.index:
                    row = result_df.loc[agent]
                    counts = row.value_counts()
                    
                    # 목표 휴무일 표시를 위해 역산 (전체공휴일 - 전역자차감)
                    target_off = scheduler.agent_targets[agent]
                    
                    stats_data.append({
                        '이름': agent,
                        '주간(주)': counts.get('주', 0),
                        '야간(야)': counts.get('야', 0),
                        '휴무(휴)': f"{counts.get('휴', 0)} / {target_off}", # 실제 / 목표
                        '비번(비)': counts.get('비', 0),
                        '면제(교)': counts.get('교', 0)
                    })
                stats_df = pd.DataFrame(stats_data).set_index('이름')
                st.write("### 📈 근무 통계 (휴무: 배정됨 / 목표)")
                st.dataframe(stats_df, use_container_width=True)

                # 2. 메인 결과 테이블
                st.write("---")
                st.subheader(f"📊 {month}월 최종 근무표")
                
                day_counts = result_df.apply(lambda col: (col == '주').sum())
                night_counts = result_df.apply(lambda col: (col == '야').sum())
                
                result_df.loc['☀️ 주간 인원'] = day_counts
                result_df.loc['🌙 야간 인원'] = night_counts
                
                styled_df = result_df.style.map(color_schedule_cells)
                st.dataframe(styled_df, use_container_width=True, height=600)
            else:
                st.error(f"오류 발생: {msg}")

if __name__ == "__main__":
    if st.session_state.step == 1:
        step1_setup()
    elif st.session_state.step == 2:
        step2_constraints()