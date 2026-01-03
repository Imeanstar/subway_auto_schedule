import pandas as pd
import numpy as np
import random
import holidays

# -----------------------------------------------------------------------------
# [상수 정의]
# -----------------------------------------------------------------------------
class ROLE:
    DAY = 0         # 주간
    NIGHT = 1       # 야간
    OFF_DUTY = 2    # 비번
    HOLIDAY = 3     # 휴무
    UNAVAILABLE = 4 # 근무불가
    DISCHARGE = 5   # 전역/교육

class ServiceScheduler:
    def __init__(self, df, year, month, preferences, discharge_dates):
        # 1. 고정값 마스크 생성
        self.fixed_mask = df.astype(str).replace(r'^\s*$', np.nan, regex=True).notna()
        
        # 2. 데이터 초기화
        self.schedule = df.copy()
        self.schedule = self.schedule.fillna("").astype(str)
        self.schedule = self.schedule.replace(r'^\s*$', np.nan, regex=True)
        self.schedule = self.schedule.apply(pd.to_numeric, errors='coerce')
        
        self.agents = self.schedule.index.tolist()
        self.dates = self.schedule.columns.tolist()
        self.total_days = len(self.dates)
        self.year = year
        self.month = month
        
        self.preferences = preferences
        self.discharge_dates = discharge_dates
        
        # 3. 목표 휴무일 계산
        self.total_month_holidays = self._count_holidays_in_range(1, self.total_days)
        self.agent_targets = {agent: self.total_month_holidays for agent in self.agents}
        
        # 4. 카운터 초기화
        self.current_off_counts = {agent: 0 for agent in self.agents}
        self.work_counts = {agent: {'day': 0, 'night': 0} for agent in self.agents}

    # =========================================================================
    # 1. 유틸리티 및 셋업
    # =========================================================================
    
    def _count_holidays_in_range(self, start_day, end_day):
        if start_day > end_day: return 0
        kr_holidays = holidays.KR(years=self.year)
        count = 0
        for day in range(start_day, end_day + 1):
            try:
                date_obj = pd.Timestamp(year=self.year, month=self.month, day=day)
                if date_obj.dayofweek >= 5 or date_obj in kr_holidays:
                    count += 1
            except ValueError:
                continue
        return count

    def _handle_discharge_dates(self):
        for agent, d_date in self.discharge_dates.items():
            if d_date and 0 < d_date < self.total_days:
                start_idx = d_date
                if start_idx < self.total_days:
                    idx = self.agents.index(agent)
                    self.schedule.iloc[idx, start_idx:] = ROLE.DISCHARGE
                    self.fixed_mask.iloc[idx, start_idx:] = True
                
                holidays_after = self._count_holidays_in_range(d_date + 1, self.total_days)
                new_target = self.total_month_holidays - holidays_after
                self.agent_targets[agent] = max(0, new_target)

    def _preprocess_fixed_data(self):
        for agent in self.agents:
            idx = self.agents.index(agent)
            for d in range(self.total_days):
                val = self.schedule.iloc[idx, d]
                
                if val == ROLE.UNAVAILABLE:
                    if d > 0 and self.schedule.iloc[idx, d-1] == ROLE.NIGHT:
                        self.schedule.iloc[idx, d] = ROLE.OFF_DUTY
                    else:
                        self.schedule.iloc[idx, d] = ROLE.HOLIDAY
                
                if val == ROLE.NIGHT:
                    if d + 1 < self.total_days:
                        if not self.fixed_mask.iloc[idx, d+1]:
                            self.schedule.iloc[idx, d+1] = ROLE.OFF_DUTY

            row = self.schedule.iloc[idx]
            self.current_off_counts[agent] = (row == ROLE.HOLIDAY).sum()
            self.work_counts[agent]['day'] = (row == ROLE.DAY).sum()
            self.work_counts[agent]['night'] = (row == ROLE.NIGHT).sum()

    # =========================================================================
    # 2. 메인 실행 (Main Loop)
    # =========================================================================

    def run(self):
        try:
            self._handle_discharge_dates()
            self._preprocess_fixed_data()
            self._fill_schedule_dynamic()
            
            # --- 후처리 파이프라인 ---
            
            # 1. 휴무 과다 -> 근무 투입 (3->0)
            self._post_process_excess_holidays()
            
            # 2. 휴무 부족 -> 근무 제외 (0->3) & 패턴 교정
            self._post_process_v5_final_fix()
            
            # 3. [Fix] 6일 패턴 교체 (조건 완화: 인원수 깨지더라도 패턴 파괴 우선)
            self._post_process_replace_6day_pattern()
            
            # 4. 최종 무결성 검사
            self._post_process_final_integrity_check()

            final_df = self._convert_to_text(self.schedule)
            return True, final_df, f"생성 완료! (목표 휴무: {self.total_month_holidays}일)"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, None, str(e)

    def _fill_schedule_dynamic(self):
        for d in range(self.total_days):
            progress = (d + 1) / self.total_days
            candidates = self._get_candidates_for_day(d)
            if not candidates: continue

            def sort_key_wrapper(role):
                return lambda ag: self._calculate_priority_score(ag, d, role, progress)

            # 1. 야간 최소 1명
            while self.schedule.iloc[:, d].eq(ROLE.NIGHT).sum() < 1:
                if not candidates: break
                candidates.sort(key=sort_key_wrapper(ROLE.NIGHT))
                self._assign_initial(candidates.pop(0), d, ROLE.NIGHT)

            # 2. 야간 최대 2명
            while self.schedule.iloc[:, d].eq(ROLE.NIGHT).sum() < 2:
                if not candidates: break
                candidates.sort(key=sort_key_wrapper(ROLE.NIGHT))
                self._assign_initial(candidates.pop(0), d, ROLE.NIGHT)

            # 3. 주간 최소 1명
            while self.schedule.iloc[:, d].eq(ROLE.DAY).sum() < 1:
                if not candidates: break
                candidates.sort(key=sort_key_wrapper(ROLE.DAY))
                best = candidates[0]
                if self._check_ya_bi_ju_violation(best, d): candidates.pop(0); continue
                self._assign_initial(candidates.pop(0), d, ROLE.DAY)

            # 4. 주간 나머지 (최대 4명)
            while self.schedule.iloc[:, d].eq(ROLE.DAY).sum() < 4:
                if not candidates: break
                candidates.sort(key=sort_key_wrapper(ROLE.DAY))
                best = candidates[0]
                if self._check_ya_bi_ju_violation(best, d): candidates.pop(0); continue
                
                tgt = self.agent_targets[best]
                is_must = (self.current_off_counts[best] >= tgt)
                usage_ratio = (self.current_off_counts[best] / tgt) if tgt > 0 else 1.0
                
                if not is_must and usage_ratio < progress:
                    candidates.pop(0); continue
                
                self._assign_initial(candidates.pop(0), d, ROLE.DAY)

            # 나머지 휴무
            for agent in self.agents:
                idx = self.agents.index(agent)
                if pd.isna(self.schedule.iloc[idx, d]):
                    self._assign_initial(agent, d, ROLE.HOLIDAY)

    # =========================================================================
    # 3. 후처리 로직 (V10 Fix)
    # =========================================================================

    def _post_process_excess_holidays(self):
        """휴무 과다 -> 근무 투입"""
        for agent in self.agents:
            idx = self.agents.index(agent)
            target = self.agent_targets[agent]
            while self.current_off_counts[agent] > target:
                off_days = [d for d in range(self.total_days) if self.schedule.iloc[idx, d] == ROLE.HOLIDAY]
                if not off_days: break
                changed = False
                for d in off_days:
                    if self.fixed_mask.iloc[idx, d]: continue
                    if self._is_valid_day_shift(idx, d):
                        self._update_cell(agent, d, ROLE.DAY)
                        changed = True; break 
                if not changed: break

    def _post_process_v5_final_fix(self):
        # 1. 야-휴 -> 야-비
        for agent in self.agents:
            idx = self.agents.index(agent)
            for d in range(self.total_days - 1):
                if self.fixed_mask.iloc[idx, d+1]: continue
                if self.schedule.iloc[idx, d] == ROLE.NIGHT and self.schedule.iloc[idx, d+1] == ROLE.HOLIDAY:
                    self._update_cell(agent, d+1, ROLE.OFF_DUTY)

        # 2. 휴무 부족 보충 (0 -> 3)
        for agent in self.agents:
            idx = self.agents.index(agent)
            target = self.agent_targets[agent]
            while self.current_off_counts[agent] < target:
                work_days = [d for d in range(self.total_days) if self.schedule.iloc[idx, d] == ROLE.DAY]
                if not work_days: break
                best_day = -1; max_workers = -1
                for d in work_days:
                    if self.fixed_mask.iloc[idx, d]: continue
                    if d > 0 and self.schedule.iloc[idx, d-1] == ROLE.NIGHT: continue
                    col_workers = self.schedule.iloc[:, d].eq(ROLE.DAY).sum()
                    if col_workers <= 1: continue 
                    if col_workers > max_workers:
                        max_workers = col_workers; best_day = d
                if best_day != -1: self._update_cell(agent, best_day, ROLE.HOLIDAY)
                else: break

    def _post_process_replace_6day_pattern(self):
        """
        [V10 Fix] 6일 패턴 교체: 야비야비야비 -> 야비휴야비휴
        수정 사항: 야간 인원이 3명이 되더라도 패턴 파괴를 1순위로 두어 강제 실행.
        """
        for agent in self.agents:
            idx = self.agents.index(agent)
            d = 0
            while d <= self.total_days - 6:
                window = self.schedule.iloc[idx, d:d+6].tolist()
                if window == [ROLE.NIGHT, ROLE.OFF_DUTY, ROLE.NIGHT, ROLE.OFF_DUTY, ROLE.NIGHT, ROLE.OFF_DUTY]:
                    # 변경 위치 고정값 체크
                    if self.fixed_mask.iloc[idx, d+2] or self.fixed_mask.iloc[idx, d+3] or \
                       self.fixed_mask.iloc[idx, d+4] or self.fixed_mask.iloc[idx, d+5]:
                        d += 1; continue
                    
                    # [V10 변경] 야간 인원 체크 제거!
                    # 설령 이날 야간이 3명이 되더라도, 사람을 살리기 위해 패턴을 끊습니다.
                    # if self.schedule.iloc[:, d+3].eq(ROLE.NIGHT).sum() >= 2: d += 1; continue

                    self._update_cell(agent, d+2, ROLE.HOLIDAY)
                    self._update_cell(agent, d+3, ROLE.NIGHT)
                    self._update_cell(agent, d+4, ROLE.OFF_DUTY)
                    self._update_cell(agent, d+5, ROLE.HOLIDAY)
                    d += 6
                else:
                    d += 1
    
    def _post_process_final_integrity_check(self):
        """최종 무결성 검사"""
        for agent in self.agents:
            idx = self.agents.index(agent)
            for d in range(self.total_days - 1):
                if self.schedule.iloc[idx, d] == ROLE.NIGHT and \
                   self.schedule.iloc[idx, d+1] != ROLE.OFF_DUTY:
                    self._update_cell(agent, d+1, ROLE.OFF_DUTY)

    # =========================================================================
    # 4. Helper Methods
    # =========================================================================

    def _get_candidates_for_day(self, d):
        candidates = []
        for agent in self.agents:
            idx = self.agents.index(agent)
            if pd.notna(self.schedule.iloc[idx, d]): continue
            if d > 0 and self.schedule.iloc[idx, d-1] == ROLE.NIGHT:
                self._assign_initial(agent, d, ROLE.OFF_DUTY); continue
            cons_work, _ = self._get_consecutive_stats(idx, d)
            if cons_work >= 5:
                self._assign_initial(agent, d, ROLE.HOLIDAY); continue
            candidates.append(agent)
        return candidates

    def _calculate_priority_score(self, agent, d, role_type, progress):
        idx = self.agents.index(agent)
        cons_work, cons_off = self._get_consecutive_stats(idx, d)
        tgt = self.agent_targets[agent]
        is_must = (self.current_off_counts[agent] >= tgt)
        p_must = 0 if is_must else 1
        
        ya_bi_ju = 0
        if role_type == ROLE.DAY and d >= 2 and self.schedule.iloc[idx, d-2] == ROLE.NIGHT: ya_bi_ju = 9999
        ya_bi_ya = 0
        if role_type == ROLE.NIGHT and d >= 2 and self.schedule.iloc[idx, d-2] == ROLE.NIGHT: ya_bi_ya = 100
        
        usage_ratio = (self.current_off_counts[agent] / tgt) if tgt > 0 else 1.0
        pacing_score = usage_ratio - progress
        off_urgency = -cons_off 
        pref = self.preferences.get(agent, "")
        p_pref = 1 
        if role_type == ROLE.NIGHT and pref == "야간 선호": p_pref = 0
        if role_type == ROLE.DAY and pref == "주간 선호": p_pref = 0

        return (ya_bi_ju, p_must, ya_bi_ya, off_urgency, -pacing_score, p_pref, random.random())

    def _assign_initial(self, agent, day_idx, value):
        idx = self.agents.index(agent)
        self.schedule.iloc[idx, day_idx] = value
        if value == ROLE.HOLIDAY: self.current_off_counts[agent] += 1
        elif value == ROLE.NIGHT:
            self.work_counts[agent]['night'] += 1
            if day_idx + 1 < self.total_days:
                if pd.isna(self.schedule.iloc[idx, day_idx+1]): self.schedule.iloc[idx, day_idx+1] = ROLE.OFF_DUTY
        elif value == ROLE.DAY: 
            self.work_counts[agent]['day'] += 1

    def _update_cell(self, agent, d, new_val):
        idx = self.agents.index(agent)
        old_val = self.schedule.iloc[idx, d]
        if old_val == ROLE.HOLIDAY: self.current_off_counts[agent] -= 1
        elif old_val == ROLE.NIGHT: self.work_counts[agent]['night'] -= 1
        elif old_val == ROLE.DAY: self.work_counts[agent]['day'] -= 1
        
        self.schedule.iloc[idx, d] = new_val
        if new_val == ROLE.HOLIDAY: self.current_off_counts[agent] += 1
        elif new_val == ROLE.NIGHT: self.work_counts[agent]['night'] += 1
        elif new_val == ROLE.DAY: self.work_counts[agent]['day'] += 1

    def _get_consecutive_stats(self, agent_idx, current_day):
        cons_work = 0; cons_off = 0
        for k in range(current_day - 1, -1, -1):
            val = self.schedule.iloc[agent_idx, k]
            if val in [ROLE.DAY, ROLE.NIGHT]: cons_work += 1
            else: break
        for k in range(current_day - 1, -1, -1):
            val = self.schedule.iloc[agent_idx, k]
            if val in [ROLE.OFF_DUTY, ROLE.HOLIDAY]: cons_off += 1
            else: break
        return cons_work, cons_off

    def _check_ya_bi_ju_violation(self, agent, d):
        idx = self.agents.index(agent)
        if d >= 2 and self.schedule.iloc[idx, d-2] == ROLE.NIGHT: return True
        return False

    def _is_valid_day_shift(self, agent_idx, d):
        if d >= 2 and self.schedule.iloc[agent_idx, d-2] == ROLE.NIGHT: return False
        if d > 0 and self.schedule.iloc[agent_idx, d-1] == ROLE.NIGHT: return False
        prev = 0
        for k in range(d - 1, -1, -1):
            if self.schedule.iloc[agent_idx, k] in [ROLE.DAY, ROLE.NIGHT]: prev += 1
            else: break
        nxt = 0
        for k in range(d + 1, self.total_days):
            if self.schedule.iloc[agent_idx, k] in [ROLE.DAY, ROLE.NIGHT]: nxt += 1
            else: break
        if (prev + 1 + nxt) > 5: return False
        return True

    def _convert_to_text(self, df):
        mapping = {0: '주', 1: '야', 2: '비', 3: '휴', 4: '불', 5: '교'}
        return df.applymap(lambda x: mapping.get(int(x), x) if pd.notna(x) else x)