import sqlite3
import math
from typing import Optional, Tuple

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class FutureTradingEnvDiscrete(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, db_file_path, start_date, end_date, input_config: dict, action_steps=3, penalty=0.003, train=True):
        # assert action_steps >= 3, 'At least 3 steps are needed'
        # self._new_normalized_position_list = np.linspace(-1, 1, action_steps)
        self._new_normalized_position_list = ['hold', 'long', 'short', 'liquidate']
        self.ticks_per_trading_day = 377  # 15:19 - 09:01 = total 378 ticks. hence, it has 378 steps of size 1/377. last tick is changed to 15:18
        self.db_file_path = db_file_path
        self.start_date = start_date
        self.end_date = end_date
        self.input_config = input_config
        self.train = train

        self.db_connection = None
        self.db_cursor = None
        self._init_db()

        # set on reset
        self.current_code = None
        self.parsed_date_list = None
        self.current_date = None
        self.current_date_idx = None
        self.initial_account_valuation = None
        self.checkpoint_account_valuation = None
        self.action_buffer = None

        # states (changes every step)
        self.current_chart = None
        self.current_price = None
        self.average_position_price = None
        self.position = None
        self.current_cash = None
        self.tick_to_close = None
        self.penalty = penalty

        chart_features = 0
        # chart_obs_dict = dict()
        for key, item in input_config['chart'].items():
            N = item['N']
            columns = item['columns']
            # chart_obs_dict[key] = spaces.Box(low=0, high=np.inf, shape=(N, len(columns)))
            chart_features += N * len(columns)
        # self.action_space = spaces.Discrete(action_steps)
        self.action_space = spaces.Discrete(4)
        # self.observation_space = spaces.Dict({
        #     'chart': spaces.Dict(chart_obs_dict),
        #     # 'account_valuation': spaces.Box(low=0, high=np.inf, shape=1),
        #     'average_position_price': spaces.Box(low=0, high=np.inf, shape=(1,)),
        #     'position': spaces.Box(low=-1, high=1, shape=(1,)),
        #     'current_price_to_current_account_valuation': spaces.Box(low=0, high=1, shape=(1,)),
        #     'tick_to_close': spaces.Box(low=0, high=1, shape=(1,)),
        # })
        # self.observation_space = spaces.Box(low=-np.finfo(np.float32).max, high=np.finfo(np.float32).max, shape=(chart_features+4, ))
        account_features = len(input_config['account'])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(chart_features + account_features,))

        self.seed()
        # self.viewer = None
        # self.state = None
        #
        # self.steps_beyond_done = None

    def copy(self):
        return self.__class__(db_file_path=self.db_file_path, start_date=self.start_date, end_date=self.end_date, input_config=self.input_config, action_steps=len(self._new_normalized_position_list), penalty=self.penalty, train=self.train)

    @property
    def current_position_valuation(self):
        return self.position * (self.current_price if self.position > 0 else (2 * self.average_position_price - self.current_price) * -1)

    @property
    def normalized_position(self):
        return self.current_position_valuation / self.current_account_valuation * (-1 if self.position < 0 else 1)

    @property
    def current_account_valuation(self):
        return self.current_cash + self.current_position_valuation

    def _init_db(self):
        self.db_connection = sqlite3.connect(self.db_file_path)
        self.db_cursor = self.db_connection.cursor()
        self.db_cursor.execute("""select name from sqlite_master where type='table'""")
        table_list = [x[0] for x in self.db_cursor.fetchall()]
        code_dict = dict()
        for table in table_list:
            asset_type, code, tick = table.split('_')
            for ic in self.input_config['chart']:
                if ic == tick:
                    if not code_dict.get(code):
                        code_dict[code] = dict()
                        code_dict[code]['asset_type'] = asset_type
                    code_dict[code][tick] = True

        self.validated_code_dict = dict()
        for code, data in code_dict.items():
            if sum(list(map(lambda data_key: data_key in self.input_config['chart'] or data_key == 'asset_type', data))) == len(self.input_config['chart']) + 1:
                self.validated_code_dict[code] = data['asset_type']


    def _normalize_price(self, price):
        return price / self.current_price - 1

    def _normalize_chart(self, chart, columns, last_price):
        if 'volume' in columns:
            raise NotImplementedError
        ohlc_indicies = [columns.index('open'), columns.index('high'), columns.index('low'), columns.index('close')]
        normalized_chart = np.zeros_like(chart, dtype=np.float32)

        # normalize o (ohlc_indicies[:1]) of first chart data with last_price
        normalized_chart[:1, ohlc_indicies[:1]] = (chart[:1, ohlc_indicies[:1]] - last_price) / last_price

        # normalize hlc (ohlc_indicies[1:]) with o (ohlc_indicies[:1])
        normalized_chart[:, ohlc_indicies[1:]] = (chart[:, ohlc_indicies[1:]] - chart[:, ohlc_indicies[:1]]) / chart[:, ohlc_indicies[:1]]
        if chart.shape[0] > 1:
            # normalize o (ohlc_indicies[:1]) with before closes (ohlc_indicies[-1:])
            normalized_chart[1:, ohlc_indicies[:1]] = (chart[1:, ohlc_indicies[:1]] - chart[:-1, ohlc_indicies[-1:]]) / chart[:-1, ohlc_indicies[-1:]]

        # assert self.current_price == chart[-1][columns.index('close')]
        # return self._normalize_price(chart)
        return normalized_chart

    def _observe(self):
        flattened_feature_list = list()
        for key, item in sorted(self.input_config['chart'].items()):
            N, columns = item['N'], item['columns']
            column_indices = list()
            for column in columns:
                column_indices.append(self.current_chart[key]['columns'].index(column))
            date_slice = slice(self.current_chart[key]['current_idx'] - (N - 1), self.current_chart[key]['current_idx'] + 1)
            chart = self.current_chart[key]['data'][date_slice, column_indices]
            last_price = self.current_chart[key]['data'][self.current_chart[key]['current_idx'] - N, self.current_chart[key]['columns'].index('close')]

            flattened_feature_list.append(self._normalize_chart(chart, columns, last_price=last_price).flatten())
            # flattened_feature_list.append(chart.flatten())

        average_position_price = self._normalize_price(self.average_position_price)
        # average_position_price = self.average_position_price

        position = self.normalized_position
        # current_price_to_current_account_valuation = self.current_price / self.current_account_valuation
        # tick_to_close = self.tick_to_close / (self.ticks_per_trading_day - 1)
        # flattened_feature_list.append([average_position_price, position, current_price_to_current_account_valuation, tick_to_close])
        flattened_feature_list.append([average_position_price, position])
        """return {
                    'chart': chart_obs_dict,
                    'average_position_price': self.average_position_price / self.current_price,
                    'position': self.normalized_position,
                    'current_price_to_current_account_valuation': self.current_price / self.current_account_valuation,
                    'tick_to_close': self.tick_to_close / (self.ticks_per_trading_day - 1)
                }"""
        return np.concatenate(flattened_feature_list, axis=0)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _trade(self, new_normalized_position) -> Tuple[float, bool]:
        old_normalized_position = self.normalized_position
        realized_profit = 0
        holded = False
        if old_normalized_position >= 0:
            if old_normalized_position < new_normalized_position:  # more long
                step = self.current_price / self.current_account_valuation

                old_position = self.position
                old_average_position_price = self.average_position_price
                old_current_cash = self.current_cash

                delta_position = int(round((new_normalized_position - old_normalized_position) / step, 8))
                assert delta_position >= 0
                contract_valuation = delta_position * self.current_price
                assert contract_valuation >= 0

                holded = delta_position == 0
                new_position = old_position + delta_position
                if new_position == 0:
                    new_average_position_price = 0
                else:
                    assert new_position > 0
                    new_average_position_price = (old_position * old_average_position_price + contract_valuation) / new_position
                new_current_cash = old_current_cash - abs(contract_valuation) * (1 + 0)  # 0 penalty for buying
                # assert new_current_cash >= 0

                self.average_position_price = new_average_position_price
                self.position = new_position
                self.current_cash = new_current_cash
            elif old_normalized_position == new_normalized_position:  # hold
                holded = True
            elif 0 <= new_normalized_position < old_normalized_position:  # liquidate long
                step = self.current_price / self.current_account_valuation

                old_position = self.position
                old_average_position_price = self.average_position_price
                old_current_cash = self.current_cash

                delta_position = int(round((new_normalized_position - old_normalized_position) / step, 8))
                assert delta_position <= 0
                contract_valuation = delta_position * self.current_price
                assert contract_valuation <= 0

                holded = delta_position == 0
                new_position = old_position + delta_position
                if new_position == 0:
                    new_average_position_price = 0
                else:
                    assert new_position > 0
                    new_average_position_price = old_average_position_price
                realization_penalty = abs(contract_valuation) * self.penalty
                new_current_cash = old_current_cash + abs(contract_valuation) - realization_penalty
                # assert new_current_cash >= 0

                self.average_position_price = new_average_position_price
                self.position = new_position
                self.current_cash = new_current_cash

                if delta_position != 0:  # < 0
                    realized_profit = (self.current_price - old_average_position_price) * abs(delta_position) - realization_penalty
            elif new_normalized_position < 0:  # liquidate all long and switch to short
                step = self.current_price / self.current_account_valuation

                old_position = self.position
                old_average_position_price = self.average_position_price
                old_current_cash = self.current_cash

                delta_position = int(round((0 - old_normalized_position) / step, 8))
                assert delta_position <= 0
                contract_valuation = delta_position * self.current_price
                assert contract_valuation <= 0

                assert old_position + delta_position == 0
                realization_penalty = abs(contract_valuation) * self.penalty
                new_current_cash = old_current_cash + abs(contract_valuation) - realization_penalty
                # assert new_current_cash >= 0

                self.average_position_price = 0
                self.position = 0
                self.current_cash = new_current_cash
                # liquidate done

                if delta_position != 0:  # < 0
                    realized_profit = (self.current_price - old_average_position_price) * abs(delta_position) - realization_penalty

                # short
                step = self.current_price / self.current_account_valuation
                delta_short_position = int(round((new_normalized_position - 0) / step, 8))
                assert delta_short_position <= 0
                short_contract_valuation = delta_short_position * self.current_price
                assert short_contract_valuation <= 0

                holded = delta_position == 0 and delta_short_position == 0

                new_position = delta_short_position
                if new_position == 0:
                    new_average_position_price = 0
                else:
                    assert new_position < 0
                    new_average_position_price = abs(short_contract_valuation / new_position)
                new_current_cash = new_current_cash - abs(short_contract_valuation) * (1 + 0)  # 0 penalty for buy
                # assert new_current_cash >= 0

                self.average_position_price = new_average_position_price
                self.position = new_position
                self.current_cash = new_current_cash
        else:
            if new_normalized_position < old_normalized_position:  # more short
                step = self.current_price / self.current_account_valuation

                old_position = self.position
                old_average_position_price = self.average_position_price
                old_current_cash = self.current_cash

                delta_position = int(round((new_normalized_position - old_normalized_position) / step, 8))
                assert delta_position <= 0
                contract_valuation = delta_position * self.current_price
                assert contract_valuation <= 0

                holded = delta_position == 0

                new_position = old_position + delta_position
                if new_position == 0:
                    new_average_position_price = 0
                else:
                    assert new_position < 0  # and old_position < 0
                    new_average_position_price = (abs(old_position) * old_average_position_price + abs(contract_valuation)) / abs(new_position)
                    # same as
                    # new_average_position_price = (old_position * old_average_position_price + contract_valuation) / new_position
                new_current_cash = old_current_cash - abs(contract_valuation) * (1 + 0)  # 0 penalty for buy
                # assert new_current_cash >= 0

                self.average_position_price = new_average_position_price
                self.position = new_position
                self.current_cash = new_current_cash
            elif old_normalized_position == new_normalized_position:  # hold
                holded = True
            elif old_normalized_position < new_normalized_position <= 0:  # liquidate short
                step = (2 * self.average_position_price - self.current_price) / self.current_account_valuation

                old_position = self.position
                old_average_position_price = self.average_position_price
                old_current_cash = self.current_cash

                delta_position = int(round((new_normalized_position - old_normalized_position) / step, 8))
                assert delta_position >= 0
                contract_valuation = delta_position * (2 * old_average_position_price - self.current_price)
                assert contract_valuation >= 0

                holded = delta_position == 0

                new_position = old_position + delta_position
                if new_position == 0:
                    new_average_position_price = 0
                else:
                    assert new_position < 0
                    new_average_position_price = old_average_position_price
                realization_penalty = abs(contract_valuation) * self.penalty
                new_current_cash = old_current_cash + abs(contract_valuation) - realization_penalty
                # assert new_current_cash >= 0

                self.average_position_price = new_average_position_price
                self.position = new_position
                self.current_cash = new_current_cash

                if delta_position != 0:  # > 0
                    realized_profit = (old_average_position_price - self.current_price) * abs(delta_position) - realization_penalty
            elif 0 < new_normalized_position:  # liquidate all short and switch to long
                step = (2 * self.average_position_price - self.current_price) / self.current_account_valuation

                old_position = self.position
                old_average_position_price = self.average_position_price
                old_current_cash = self.current_cash

                delta_position = int(round((0 - old_normalized_position) / step, 8))
                assert delta_position >= 0
                contract_valuation = delta_position * (2 * old_average_position_price - self.current_price)
                assert contract_valuation >= 0

                assert old_position + delta_position == 0
                realization_penalty = abs(contract_valuation) * self.penalty
                new_current_cash = old_current_cash + abs(contract_valuation) - realization_penalty
                # assert new_current_cash >= 0

                self.average_position_price = 0
                self.position = 0
                self.current_cash = new_current_cash
                # liquidate done

                if delta_position != 0:  # > 0
                    realized_profit = (old_average_position_price - self.current_price) * abs(delta_position) - realization_penalty

                # long
                long_step = self.current_price / (self.current_account_valuation)

                delta_long_position = int(round((new_normalized_position - 0) / long_step, 8))
                assert delta_long_position >= 0
                long_contract_valuation = delta_long_position * self.current_price
                assert long_contract_valuation >= 0

                holded = delta_position == 0 and delta_long_position == 0
                assert not holded, 'Variable holded can not be True in this case'

                new_position = delta_long_position
                if new_position == 0:
                    new_average_position_price = 0
                else:
                    assert new_position > 0
                    new_average_position_price = abs(long_contract_valuation / new_position)
                new_current_cash = new_current_cash - abs(long_contract_valuation) * (1 + 0)  # 0 penalty for buy
                # assert new_current_cash >= 0

                self.average_position_price = new_average_position_price
                self.position = new_position
                self.current_cash = new_current_cash
        return realized_profit, holded

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.action_buffer.append(action)
        if False:
            new_normalized_position = self._new_normalized_position_list[action]
        else:
            contract_step = 0.2
            if action == 0:  # hold
                new_normalized_position = self.normalized_position
            elif action == 1:  # long 50%
                new_normalized_position = np.clip(self.normalized_position + contract_step, a_min=-1., a_max=1.)
            elif action == 2:  # short 50%
                new_normalized_position = np.clip(self.normalized_position - contract_step, a_min=-1., a_max=1.)
            elif action == 3:  # liquidate all
                new_normalized_position = 0

        old_position = self.position
        old_current_account_valuation = self.current_account_valuation
        realized_profit, holded = self._trade(new_normalized_position=new_normalized_position)

        assert (holded and (realized_profit == 0)) or not holded, 'Something is wrong'

        reward = 0

        use_reward_r = True
        allow_negative_r = True
        weight_positive_r = None

        use_reward_p = False
        allow_negative_p = False
        weight_positive_p = 10

        use_hold_bonus = False

        use_reward_d = False
        allow_negative_d = True

        use_no_action_penalty = False
        no_action_penalty_n = 2

        use_reward_ritter = True

        assert not use_reward_d or not use_reward_r, 'done reward and realized reward cannot be used together'

        if use_reward_ritter and False:
            assert not use_reward_r and not use_reward_p and not use_hold_bonus and not use_reward_d, 'Reward Ritter must be used solely'

        if use_reward_r:  # realized reward
            # if not self.train:
            #     print(f'realized profit: {realized_profit}')
            if realized_profit > 0 or allow_negative_r:  # allow negative?
                realized_reward = realized_profit / self.initial_account_valuation
                if weight_positive_r is not None:
                    realized_reward = realized_reward * weight_positive_r if realized_reward > 0 else realized_reward
                reward += realized_reward


        if use_no_action_penalty:
            last_n_actions = np.asarray(self.action_buffer[-no_action_penalty_n:])
            if len(last_n_actions) >= no_action_penalty_n and (last_n_actions == last_n_actions[-1]).all():
                reward -= 0.001 / 377

        done = self.tick_to_close == 0
        if done:
            realized_profit, _ = self._trade(new_normalized_position=0.)  # liquidate all
            if use_reward_d:  # done reward
                done_reward = (self.current_account_valuation / self.initial_account_valuation) - 1.0
                if done_reward > 0 or allow_negative_d:
                    reward += done_reward
            elif use_reward_r:  # realized reward at last step
                if realized_profit > 0 or allow_negative_r:  # allow negative?
                    reward += realized_profit / self.initial_account_valuation
        if use_reward_ritter:
            delta_v_t = (self.current_account_valuation - old_current_account_valuation) / self.initial_account_valuation
            k = 0
            reward_ritter = delta_v_t - k / 2 * (delta_v_t ** 2)
            reward += reward_ritter

        # reward = 10 ** (reward) - (2 ** -0.) * 0

        return self._observe(), reward, done, {}

    def _get_chart_data(self, code, tick, start_date_time, end_date_time):
        self.db_cursor.execute(f'select * from {self.validated_code_dict[code]}_{code}_{tick} where date >= ? and date <= ? order by date', (start_date_time, end_date_time))
        data = np.asarray(self.db_cursor.fetchall())
        columns = [x[0] for x in self.db_cursor.description]
        return data, columns


    def _add_time_to_date(self, date: int, time: int) -> int:
        d, t = f'{date:08}', f'{time:06}'
        assert len(f'{date:08}') == 8, 'date must be 8 digits integer'
        assert len(f'{time:06}') == 6, 'time must be 6 digits integer'
        return int(d + t)

    def _get_num_contract(self):
        if self.position > 0:
            return self.current_account_valuation * self.position / self.current_price
        else:
            return self.current_account_valuation * self.position / (2 * self.average_position_price - self.current_price)

    def get_daily_profit(self):
        return self.current_account_valuation / self.initial_account_valuation - 1

    def print_profit(self):
        if self.tick_to_close == 0:
            print(f'Current profit: {self.get_daily_profit() * 100}%', end='\t')

    def reset(self, code: Optional[str] = None, date_idx: Optional[int] = None):
        if self.tick_to_close is not None and self.train:
            self.print_profit()
        self.action_buffer = []
        # choose code from db and date between [self.start_date, self.end_date]
        if self.db_cursor is None:
            self.db_cursor = self.db_connection.cursor()

        _past_code = self.current_code
        while True:
            if code is None:
                self.current_code = np.random.choice(list(self.validated_code_dict))
            else:
                assert code in self.validated_code_dict, 'Given code is not in database'
                self.current_code = code
            if _past_code != self.current_code:
                self._parse_date_from_1min_candle()


            if self.current_chart is None:
                assert code is None and date_idx is None, f'Given code and date_idx is wrong. code: {code}, date_idx: {date_idx}'
                continue
            else:
                break

        use_random_account = True
        if self.train and use_random_account:
            self.average_position_price = self.current_price + int(np.random.uniform(-20, 20)) * int(self.current_price * 0.001)
            self.position = int(np.random.normal(loc=0., scale=0.5) * (10000000 // self.current_price))
            self.current_cash = int(np.random.uniform(1, 10000)) * 1000
            self.checkpoint_account_valuation = self.initial_account_valuation = self.current_account_valuation
        else:  # cash 100%
            self.average_position_price = 0
            self.position = 0
            self.current_cash = int(np.random.uniform(500, 10000)) * 1000
            # self.current_cash = 10000000
            self.checkpoint_account_valuation = self.initial_account_valuation = self.current_account_valuation
            self.current_cash = self.initial_account_valuation

        return self._observe()

    def render(self, mode='human'):
        print('\n------------- RENDER START -------------')
        print(f'code: {self.current_code}, date: {self.current_date} profit: {self.get_daily_profit() * 100:.3f}%')
        print([self._new_normalized_position_list[a] for a in self.action_buffer])
        print('------------- RENDER DONE -------------\n')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


