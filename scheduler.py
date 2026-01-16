import pandas as pd
import numpy as np
import holidays
from ortools.sat.python import cp_model

# -----------------------------------------------------------------------------
# [상수 정의]
# -----------------------------------------------------------------------------
class ROLE:
    DAY = 0         # 주간
    NIGHT = 1       # 야간
    OFF_DUTY = 2    # 비번
    HOLIDAY = 3     # 휴무
    UNAVAILABLE = 4 # 근무불가 (주/야 모두 안됨)
    DISCHARGE = 5   # 전역/교육
    DAY_UNAVAILABLE = 6   # 주간만 불가 (야간/비번/휴무 가능)
    NIGHT_UNAVAILABLE = 7  # 야간만 불가 (주간/비번/휴무 가능)

class ServiceScheduler:
    def __init__(self, df, year, month, preferences, discharge_dates):
        # 1. 고정값 마스크 생성
        self.fixed_mask = df.astype(str).replace(r'^\s*$', np.nan, regex=True).notna()
        self.schedule = df.copy()
        
        # 전처리: 숫자 변환
        self.schedule = self.schedule.fillna("").astype(str)
        self.schedule = self.schedule.replace(r'^\s*$', np.nan, regex=True)
        self.schedule = self.schedule.apply(pd.to_numeric, errors='coerce')
        
        self.agents = self.schedule.index.tolist()
        self.dates = self.schedule.columns.tolist()
        self.num_agents = len(self.agents)
        self.num_days = len(self.dates)
        
        self.year = year
        self.month = month
        self.preferences = preferences
        self.discharge_dates = discharge_dates
        
        # 2. 목표 휴무일 계산
        self.total_month_holidays = self._count_holidays_in_range(1, self.num_days)
        self.agent_targets = {agent: self.total_month_holidays for agent in self.agents}
        
        self._handle_discharge_logic()

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

    def _handle_discharge_logic(self):
        for n, agent in enumerate(self.agents):
            d_date = self.discharge_dates.get(agent, 0)
            if d_date and 0 < d_date < self.num_days:
                self.schedule.iloc[n, d_date:] = ROLE.DISCHARGE
                self.fixed_mask.iloc[n, d_date:] = True
                holidays_after = self._count_holidays_in_range(d_date + 1, self.num_days)
                new_target = self.total_month_holidays - holidays_after
                self.agent_targets[agent] = max(0, new_target)

    def run(self):
        model = cp_model.CpModel()
        shifts = {}
        for n in range(self.num_agents):
            for d in range(self.num_days):
                shifts[(n, d)] = model.NewIntVar(0, 5, f'shift_n{n}_d{d}')

        # ---------------------------------------------------------------------
        # 2. Hard Constraints (절대 규칙)
        # ---------------------------------------------------------------------

        # (A) 고정값 및 특수 입력(4, 6, 7) 처리
        for n in range(self.num_agents):
            for d in range(self.num_days):
                if self.fixed_mask.iloc[n, d]:
                    val = int(self.schedule.iloc[n, d])
                    
                    # 4: 전체 불가 -> 비(2) or 휴(3)
                    if val == ROLE.UNAVAILABLE:
                        allowed = [ROLE.OFF_DUTY, ROLE.HOLIDAY]
                        if d == 0: allowed = [ROLE.HOLIDAY] # 첫날 비번 금지
                        model.AddLinearExpressionInDomain(shifts[(n, d)], cp_model.Domain.FromValues(allowed))
                    
                    # 6: 주간 불가 -> 야(1), 비(2), 휴(3)
                    elif val == ROLE.DAY_UNAVAILABLE:
                        allowed = [ROLE.NIGHT, ROLE.OFF_DUTY, ROLE.HOLIDAY]
                        if d == 0: allowed = [ROLE.NIGHT, ROLE.HOLIDAY] # 첫날 비번 금지
                        model.AddLinearExpressionInDomain(shifts[(n, d)], cp_model.Domain.FromValues(allowed))
                    
                    # 7: 야간 불가 -> 주(0), 비(2), 휴(3)
                    elif val == ROLE.NIGHT_UNAVAILABLE:
                        allowed = [ROLE.DAY, ROLE.OFF_DUTY, ROLE.HOLIDAY]
                        if d == 0: allowed = [ROLE.DAY, ROLE.HOLIDAY] # 첫날 비번 금지
                        model.AddLinearExpressionInDomain(shifts[(n, d)], cp_model.Domain.FromValues(allowed))
                    
                    else:
                        # 0, 1, 2, 3, 5 고정
                        model.Add(shifts[(n, d)] == val)
                
                else:
                    # 빈칸: 불(4), 교(5) 제외한 기본 근무조
                    allowed = [ROLE.DAY, ROLE.NIGHT, ROLE.OFF_DUTY, ROLE.HOLIDAY]
                    if d == 0: allowed = [ROLE.DAY, ROLE.NIGHT, ROLE.HOLIDAY] # 첫날 비번 금지
                    model.AddLinearExpressionInDomain(shifts[(n, d)], cp_model.Domain.FromValues(allowed))

        # (B) 근무 인원 제약 (주간/야간 1~2명)
        for d in range(self.num_days):
            day_workers = []
            night_workers = []
            for n in range(self.num_agents):
                is_day = model.NewBoolVar(f'is_day_{n}_{d}')
                model.Add(shifts[(n, d)] == ROLE.DAY).OnlyEnforceIf(is_day)
                model.Add(shifts[(n, d)] != ROLE.DAY).OnlyEnforceIf(is_day.Not())
                day_workers.append(is_day)
                
                is_night = model.NewBoolVar(f'is_night_{n}_{d}')
                model.Add(shifts[(n, d)] == ROLE.NIGHT).OnlyEnforceIf(is_night)
                model.Add(shifts[(n, d)] != ROLE.NIGHT).OnlyEnforceIf(is_night.Not())
                night_workers.append(is_night)

            model.Add(sum(day_workers) >= 1)
            model.Add(sum(day_workers) <= 2) 
            model.Add(sum(night_workers) >= 1)
            model.Add(sum(night_workers) <= 2)

        # (C) 근무 패턴 제약
        for n in range(self.num_agents):
            for d in range(self.num_days):
                is_night = model.NewBoolVar(f'is_n_{n}_{d}')
                model.Add(shifts[(n, d)] == ROLE.NIGHT).OnlyEnforceIf(is_night)
                model.Add(shifts[(n, d)] != ROLE.NIGHT).OnlyEnforceIf(is_night.Not())
                
                is_off = model.NewBoolVar(f'is_off_{n}_{d}')
                model.Add(shifts[(n, d)] == ROLE.OFF_DUTY).OnlyEnforceIf(is_off)
                model.Add(shifts[(n, d)] != ROLE.OFF_DUTY).OnlyEnforceIf(is_off.Not())

                # 1. 야간 -> 다음날 비번
                if d < self.num_days - 1:
                    model.Add(shifts[(n, d+1)] == ROLE.OFF_DUTY).OnlyEnforceIf(is_night)

                # 2. 비번 -> 전날 야간
                if d > 0:
                    model.Add(shifts[(n, d-1)] == ROLE.NIGHT).OnlyEnforceIf(is_off)

                # 3. 야-비-주 금지
                if d < self.num_days - 2:
                    model.Add(shifts[(n, d+2)] != ROLE.DAY).OnlyEnforceIf(is_night)

        # (D) 5일 연속 근무 제한
        for n in range(self.num_agents):
            for d in range(self.num_days - 5):
                is_works = []
                for k in range(6):
                    is_work = model.NewBoolVar(f'work_{n}_{d+k}')
                    model.AddLinearExpressionInDomain(shifts[(n, d+k)], cp_model.Domain.FromValues([ROLE.DAY, ROLE.NIGHT])).OnlyEnforceIf(is_work)
                    model.AddLinearExpressionInDomain(shifts[(n, d+k)], cp_model.Domain.FromIntervals([[2, 5]])).OnlyEnforceIf(is_work.Not())
                    is_works.append(is_work)
                model.Add(sum(is_works) <= 5)

        # (E) 휴무 갯수 엄수
        for n in range(self.num_agents):
            agent_name = self.agents[n]
            target = self.agent_targets[agent_name]
            is_holidays = []
            for d in range(self.num_days):
                is_hol = model.NewBoolVar(f'is_hol_{n}_{d}')
                model.Add(shifts[(n, d)] == ROLE.HOLIDAY).OnlyEnforceIf(is_hol)
                model.Add(shifts[(n, d)] != ROLE.HOLIDAY).OnlyEnforceIf(is_hol.Not())
                is_holidays.append(is_hol)
            model.Add(sum(is_holidays) == target)

        # ---------------------------------------------------------------------
        # 3. Soft Constraints (Objectives)
        # ---------------------------------------------------------------------
        penalties = []

        # (A) 야간 우선 배정 (주간에 10점 벌점)
        for n in range(self.num_agents):
            for d in range(self.num_days):
                is_day = model.NewBoolVar(f'penalty_day_{n}_{d}')
                model.Add(shifts[(n, d)] == ROLE.DAY).OnlyEnforceIf(is_day)
                model.Add(shifts[(n, d)] != ROLE.DAY).OnlyEnforceIf(is_day.Not())
                penalties.append(is_day * 10)

        # (B) 야-비-야-비 패턴 방지
        for n in range(self.num_agents):
            for d in range(self.num_days - 2):
                is_bad_pattern = model.NewBoolVar(f'bad_pat_{n}_{d}')
                s1, s2, s3 = model.NewBoolVar(f's1_{n}_{d}'), model.NewBoolVar(f's2_{n}_{d}'), model.NewBoolVar(f's3_{n}_{d}')
                model.Add(shifts[(n, d)] == ROLE.NIGHT).OnlyEnforceIf(s1)
                model.Add(shifts[(n, d)] != ROLE.NIGHT).OnlyEnforceIf(s1.Not())
                model.Add(shifts[(n, d+1)] == ROLE.OFF_DUTY).OnlyEnforceIf(s2)
                model.Add(shifts[(n, d+1)] != ROLE.OFF_DUTY).OnlyEnforceIf(s2.Not())
                model.Add(shifts[(n, d+2)] == ROLE.NIGHT).OnlyEnforceIf(s3)
                model.Add(shifts[(n, d+2)] != ROLE.NIGHT).OnlyEnforceIf(s3.Not())
                model.AddBoolAnd([s1, s2, s3]).OnlyEnforceIf(is_bad_pattern)
                model.AddBoolOr([s1.Not(), s2.Not(), s3.Not()]).OnlyEnforceIf(is_bad_pattern.Not())
                penalties.append(is_bad_pattern * 100)

        # (C) 선호도 반영
        for n, agent in enumerate(self.agents):
            pref = self.preferences.get(agent, "")
            for d in range(self.num_days):
                if pref == "주간 선호":
                    is_day = model.NewBoolVar(f'pref_d_{n}_{d}')
                    model.Add(shifts[(n, d)] == ROLE.DAY).OnlyEnforceIf(is_day)
                    model.Add(shifts[(n, d)] != ROLE.DAY).OnlyEnforceIf(is_day.Not())
                    penalties.append(is_day.Not() * 5)
                elif pref == "야간 선호":
                    is_night = model.NewBoolVar(f'pref_n_{n}_{d}')
                    model.Add(shifts[(n, d)] == ROLE.NIGHT).OnlyEnforceIf(is_night)
                    model.Add(shifts[(n, d)] != ROLE.NIGHT).OnlyEnforceIf(is_night.Not())
                    penalties.append(is_night.Not() * 5)

        # ---------------------------------------------------------------------
        # 4. Solve
        # ---------------------------------------------------------------------
        model.Minimize(sum(penalties))
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            for n in range(self.num_agents):
                for d in range(self.num_days):
                    self.schedule.iloc[n, d] = solver.Value(shifts[(n, d)])
            return True, self._convert_to_text(self.schedule), f"최적화 완료! (주간/야간 불가 반영)"
        else:
            return False, None, "조건을 만족하는 해를 찾을 수 없습니다. (입력된 근무불가 조건이 너무 많을 수 있습니다.)"

    def _convert_to_text(self, df):
        mapping = {0: '주', 1: '야', 2: '비', 3: '휴', 4: '불', 5: '교'}
        return df.applymap(lambda x: mapping.get(int(x), x) if pd.notna(x) else x)