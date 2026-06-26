import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scil
from joblib import Parallel, delayed

class AmericanPutSimulator():

    def __init__(
        self,
        initial_stock_price : float,
        total_time : int,
        maturity : int ,
        risk_free_rate_total : float,
        volatility : float,
        real_probability : float,
        number_simulations : int,
        number_observations : int,
        strike_price : float
    ):
        """
        Initializes all variables needed in the simulation.

        Args:
            initial_stock_price: the stock price at time zero. (float)
            total_time: the calendar time to the maturity. (int)
            maturity: the last time step, i.e., the options maturity (int)
            risk_free_rate_total: the risk-free interest rate from the total time. (float)
            volatility: the volatility of the stock price. (float)
            real_probability: the probability of the up-movement of the stock. (float)
            number_simulations: the number of times the simulation is repeated. (int)
            number_observations: the number of observations of logarithmic returns
            used to estimate the stock price volatilty (int)
            strike_price: the options strike price (float)

        """
        self.intrinsic_value = lambda x : (strike_price-x).clip(0) 
        self.initial_stock_price = initial_stock_price 

        self.number_simulations = number_simulations 
        self.time_interval = total_time/maturity
        self.maturity = maturity 

        self.number_obervations = number_observations 
        self.volatility = volatility 

        self.up_factor = np.exp(volatility*np.sqrt(self.time_interval)) 
        self.down_factor = np.exp(-volatility*np.sqrt(self.time_interval)) 
        self.risk_free_rate = np.exp(risk_free_rate_total*self.time_interval) 
    
        self.risk_neutral_p = (
            self.risk_free_rate-self.down_factor
        )/(self.up_factor-self.down_factor)

        self.risk_neutral_q = 1-self.risk_neutral_p 
        self.real_probability = real_probability
        

    def get_stock_price_tree(self,
      up_factor : float = None,
      down_factor : float= None
      ) -> np.ndarray:
        """
        Builds the stock price tree into a np.ndarray such that it is indexed
        S[j,n], where n is the time step and j is the number up movements of 
        the stock in that part of the tree.

        Args:
            up-factor, float, which is the up-factor used to calculate the
            stock price tree
            down-factor, float, which is the down-factor used to calculate the
            stock price tree

        Returns:
            np.ndarray, which contains the stock price tree.

        """
        if up_factor is None and down_factor is None:
            number_of_steps = self.maturity + 1 

            # First, a matrix of up-factors
            up_factor_matrix = np.rot90( np.vander(np.full(
                number_of_steps, self.up_factor
                ),number_of_steps),k=1, axes=(0, 1))

            # Then a matrix of down-factors
            column = np.zeros(number_of_steps)
            column[0] = 1
            down_factor_matrix = scil.toeplitz(
                column, self.down_factor ** np.arange(number_of_steps))

            return (up_factor_matrix * down_factor_matrix * self.initial_stock_price)
        else:
            number_of_steps = self.maturity + 1 

            # First, a matrix of up-factors
            up_factor_matrix = np.rot90( np.vander(np.full(
                number_of_steps, up_factor
                ),number_of_steps),k=1, axes=(0, 1))

            # Then a matrix of down-factors
            column = np.zeros(number_of_steps)
            column[0] = 1
            down_factor_matrix = scil.toeplitz(
                column, down_factor ** np.arange(number_of_steps))

            return (up_factor_matrix * down_factor_matrix * self.initial_stock_price)

    
    def get_option_price_tree(
        self,
        up_factor : float = None,
        down_factor : float= None
        ) -> np.ndarray:
        """
        Builds the option price tree into an Numpu array such that it is indexed
        V[j,n], where n is the time step and j is the number up movements of the 
        stock in that part of the tree.

        Args:
            up-factor, float, which is the up-factor used to calculate the
            option price tree
            down-factor, float, which is the down-factor used to calculate the
            option price tree

        Returns:
            np.ndarray, which contains the option price tree.
        
        """
        if up_factor is None and down_factor is None:
            stock_price_tree = self.get_stock_price_tree()

            result = np.zeros((self.maturity+1,self.maturity+1))

            for inverse_column_index in range(self.maturity+1):
                for row_index in range(self.maturity+1):

                    column_index = self.maturity-inverse_column_index

                    if row_index > column_index:
                        continue

                    if column_index==self.maturity:
                        result[row_index,column_index] = max(self.intrinsic_value(
                            stock_price_tree[row_index,column_index]
                            ),0)
                    else:
                        result[row_index,column_index] = max(
                            self.intrinsic_value(
                                stock_price_tree[row_index,column_index]
                                ),
                            (
                            self.risk_neutral_p*result[
                                row_index+1,column_index+1
                                ]
                            +self.risk_neutral_q*result[
                                row_index,column_index+1
                                ]
                            )/(self.risk_free_rate)
                            )

            return result
        else:
            stock_price_tree = self.get_stock_price_tree(up_factor,down_factor)

            result = np.zeros((self.maturity+1,self.maturity+1))

            risk_neutral_p = (
            self.risk_free_rate-down_factor
            )/(up_factor-down_factor)
            risk_neutral_q = 1 - risk_neutral_p

            for inverse_column_index in range(self.maturity+1):
                for row_index in range(self.maturity+1):

                    column_index = self.maturity-inverse_column_index

                    if row_index > column_index:
                        continue

                    if column_index==self.maturity:
                        result[row_index,column_index] = max(self.intrinsic_value(
                            stock_price_tree[row_index,column_index]
                            ),0)
                    else:
                        result[row_index,column_index] = max(
                            self.intrinsic_value(
                                stock_price_tree[row_index,column_index]
                                ),
                            (
                            risk_neutral_p*result[
                                row_index+1,column_index+1
                                ]
                            +risk_neutral_q*result[
                                row_index,column_index+1
                                ]
                            )/(self.risk_free_rate)
                            )
            return result

    def simulate_price(self,seed) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulates one path of the stock price.
        
        Returns:
            tuple[np.ndarray, np.ndarray], where:
            1. element is a np.ndarray and contains the single simulated stock
            price path
            2. element is a np.ndarray and contains the number of up-movements
            of the simulated stock price till the current time step.

        """
        result_price = np.zeros(self.maturity+1)
        result_number_up_factors = np.zeros(self.maturity+1)
        
        for i in range(self.maturity+1):
            random_choice = np.random.choice(np.array([0, 1]),
            p=np.array([1- self.real_probability,self.real_probability])
            )
            if random_choice==1:
                if i == 0:
                    result_price[i] = self.initial_stock_price
                    result_number_up_factors[i] = 0
                else:
                    result_price[i] = result_price[i-1]*self.up_factor
                    result_number_up_factors[i] = result_number_up_factors[i-1]+1
            elif random_choice==0:
                if i == 0:
                    result_price[i]=self.initial_stock_price
                    result_number_up_factors[i] = 0
                else:
                    result_price[i] = result_price[i-1]*self.down_factor
                    result_number_up_factors[i] = result_number_up_factors[i-1]
        return result_price, result_number_up_factors

    def get_delta_hedge(self, seed) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    ]:
        """
        Samples one path of the stock price and then calculates 
        the delta hedge for that path using the error modified up- and
        down-factors.

        Args:
            seed: takes in the seed number to initialize np.random.seed (float)

        Returns:
            tuple[np.ndarray,float,np.ndarray,np.ndarray,np.ndarray] , where:
                1. element is a np.ndarray which contains the delta hedge.
                2. element is a np.ndarray which contains the simulated stock
                price.
                3. element is a np.ndarray which contains the number of 
                up-factors in the stock price path up to the current time step.


        """
        
        np.random.seed(seed)
        volatility_with_error = np.sqrt(
            np.random.chisquare(
                self.number_obervations-1
                )/(self.number_obervations-1)
        )*self.volatility
        up_factor_with_error = np.exp(volatility_with_error *np.sqrt(self.time_interval)) 
        down_factor_with_error = 1/up_factor_with_error

        stock_price, number_u_factors = self.simulate_price(seed)
        option_tree = self.get_option_price_tree(
            up_factor_with_error,
            down_factor_with_error
            )

        result_delta_hedge = np.zeros(self.maturity+1)

        for n in range(self.maturity):
            result_delta_hedge[n] = (option_tree[int(number_u_factors[n])+1, n+1]
            -option_tree[int(number_u_factors[n]), n+1])/(
                (
                    up_factor_with_error-down_factor_with_error
                    )*stock_price[n]
            )
        return result_delta_hedge, stock_price, number_u_factors

    def get_single_pnl_run(self, seed) -> tuple[float, float]:
        """

            Simulates one path of the stock price then calculates the delta hedge
            using the get_delta_hedge method. After this it simulates the PnL of
            the bank on the simulates path by assuming the behaviour of the 
            option's owner. Particularly, it assumes that the option is exercised
            if the intrinsic value is larger than the continuation value, i.e,
            if the cash flow is positive.

            Args:
                seed: takes in the seed number to initialize np.random.seed (float)

            Returns:
                tuple[float, float], where:
                1. element is the PnL of the bank, without the hedge.
                2. element is the PnL of the bank, with the hedge.

        """
        option_tree = self.get_option_price_tree()
        initial_option_price = option_tree[0,0]
        delta, stock_price, number_u_factors = (
            self.get_delta_hedge(seed)
            )
        replication_value = np.zeros(self.maturity+1)
        replication_value[0] = initial_option_price
        execution_index = -1

        for time_step in range(1,self.maturity+1):
            
            if time_step != self.maturity:
                cash_flow = option_tree[int(number_u_factors[time_step]), time_step]-(
                    self.risk_neutral_p*option_tree[int(
                        number_u_factors[time_step]
                        )+1, time_step+1]
                    +self.risk_neutral_q*option_tree[
                        int(number_u_factors[time_step]), time_step+1
                        ]
                )/(self.risk_free_rate)
            else:
                cash_flow = self.intrinsic_value(stock_price[time_step])

            if cash_flow > 0:
                execution_index = time_step
                portfolio_at_execution = (
                    delta[time_step-1]*stock_price[time_step]
                )+(
                    self.risk_free_rate
                    )*(
                    replication_value[time_step-1]
                    -delta[time_step-1]*stock_price[time_step-1]
                    )
                break

            replication_value[time_step] = (
                delta[time_step-1]*stock_price[time_step]
            )+(
                self.risk_free_rate
            )*(
                replication_value[time_step-1]
                -delta[time_step-1]*stock_price[time_step-1]
                )

        if execution_index == -1:
            no_hedge_pnl = initial_option_price 
            hedge_pnl = replication_value[-1]
        else:
            no_hedge_pnl = initial_option_price - self.intrinsic_value(
                stock_price[execution_index]
                )
            hedge_pnl = portfolio_at_execution - self.intrinsic_value(
                stock_price[execution_index]
                )

        return no_hedge_pnl, hedge_pnl

    def get_pnl_histograms(self):
        """
        Runs the get_single_pnl_run method number_simulation times using 
        Joblib's parallel computing functionality. Then plots histograms of
        these using Matplotlib.

        """
        result_no_hedge = []
        result_hedge = []
        np.random.seed(0)
        seeds = np.random.choice(range(1,self.number_simulations*10),
         size=self.number_simulations, replace=False)

    
        def task(seed):
            run_no_h, run_h = self.get_single_pnl_run(seed)
            return (run_no_h, run_h)
        
        results_list = Parallel(n_jobs=-1)(
            delayed(task)(seeds[i]) for i in range(self.number_simulations)
            )
        
        result_no_hedge, result_hedge = zip(*results_list)

        result_no_hedge = np.array(result_no_hedge)
        result_hedge = np.array(result_hedge)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(
            f'Frequency of PnL with $k=${self.number_obervations}',
            fontsize=12
        )

        ax1.hist(result_no_hedge, bins='auto',
        color='orange', alpha=0.5, log=True
        )

        ax1.axvline(result_no_hedge.mean(), color='orange',
        linestyle='dashed', linewidth=1, label='mean'
        )

        ax1.axvline(np.percentile(result_no_hedge,10), color='red',
        linestyle='dashed', linewidth=1, label='P10/P90'
        )

        ax1.axvline(np.percentile(result_no_hedge,90), color='red',
        linestyle='dashed', linewidth=1
        )

        ax2.hist(result_hedge, bins='auto',
        color='blue', alpha=0.5, log=True
        )

        ax2.axvline(result_hedge.mean(), color='blue',
        linestyle='dashed', linewidth=1, label='mean'
        )

        ax2.axvline(np.percentile(result_hedge,10), color='red',
        linestyle='dashed', linewidth=1, label='P10/P90'
        )

        ax2.axvline(np.percentile(result_hedge,90), color='red',
        linestyle='dashed', linewidth=1
        )


        ax1.set(xlabel='PnL (€)', ylabel='Frequency',
        title='PnL without the hedge'
        )
        ax1.legend( )

        ax2.set(xlabel='PnL (€)', ylabel='Frequency',
        title=f'PnL with the hedge and $k=${self.number_obervations}'
        )
        ax2.legend( )
        plt.tight_layout()
        fig.tight_layout()



simulation = AmericanPutSimulator(
    initial_stock_price=100,
    total_time=1,
    maturity=365,
    risk_free_rate_total=0.05,
    volatility=0.25,
    real_probability=0.505,
    number_simulations=5000 ,
    number_observations=10, 
    strike_price=100
    )

simulation.get_pnl_histograms()

plt.show()