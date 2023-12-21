import pandas as pd
import statsmodels.api as sm
from typing import List


class FeatureReduction(object):
    def __init__(self):
        pass

    @staticmethod
    def forward_selection(
        data: pd.DataFrame, target: pd.Series, significance_level: float = 0.05
    ) -> List[str]:  # 9 pts
        """
        Args:
            data: (pandas data frame) contains the feature matrix
            target: (pandas series) represents target feature to search to generate significant features
            significance_level: (float) threshold to reject the null hypothesis
        Return:
            forward_list: (python list) contains significant features. Each feature
            name is a string
        """
        initial_features = data.columns.tolist()
        forward_list = []
        while len(initial_features) > 0:
            rem = list(set(initial_features) - set(forward_list))
            pvals = pd.Series(index=rem,dtype='float64')
            for column in rem:
                model = sm.OLS(target, sm.add_constant(data[forward_list +[column]])).fit()
                pvals[column] = model.pvalues[column]
                min_pval = pvals.min()
            if min_pval < significance_level:
                forward_list.append(pvals.idxmin())
            else:
                break
        return forward_list

    @staticmethod
    def backward_elimination(
        data: pd.DataFrame, target: pd.Series, significance_level: float = 0.05
    ) -> List[str]:  # 9 pts
        """
        Args:
            data: (pandas data frame) contains the feature matrix
            target: (pandas series) represents target feature to search to generate significant features
            significance_level: (float) threshold to reject the null hypothesis
        Return:
            backward_list: (python list) contains significant features. Each feature
            name is a string
        """
        backward_list = list(data.columns)
        while len(backward_list) > 0:
            X = data[backward_list]
            X = sm.add_constant(X)
            model = sm.OLS(target, X).fit()
            pvals = model.pvalues[1:]
            #Max pval
            max_pval = pvals[pvals.idxmax()]
            if max_pval >= significance_level:
                backward_list.remove(pvals.idxmax())
            else:
                break
        return backward_list
