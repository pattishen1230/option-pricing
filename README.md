<div id="top" align="center">
<p align="center">
  <strong>
    <h2 align="center">American Option Pricing using Self-Attention GRU and Shapley Value Interpretation</h2>
  </strong>
</p>
</div>

<br/>

> **[Working Paper Link](https://arxiv.org/abs/2310.12500)**
>
> [Yanhui Shen](https://pattishen1230.github.io/)
>
> Georgia State University 
> <br/>


## Abstract
Options, serving as a crucial financial instrument, are used by investors to manage and mitigate their investment risks within the securities market. 
Precisely predicting the present price of an option enables investors to make informed and efficient decisions.
In this paper, we propose a machine learning method for forecasting the prices of SPY (ETF) option based on gated recurrent unit (GRU) and self-attention mechanism. 
We first partitioned the raw dataset into 15 subsets according to moneyness and days to maturity criteria. For each subset, we matched the corresponding U.S. government bond rates and Implied Volatility Indices. This segmentation allows for a more insightful exploration of the impacts of risk-free rates and underlying volatility on option pricing. 
Next, we built four different machine learning models, including multilayer perceptron (MLP), long short-term memory (LSTM), self-attention LSTM, and self-attention GRU in comparison to the traditional binomial model.
The empirical result shows that self-attention GRU with historical data outperforms other models due to its ability to capture complex temporal dependencies and leverage the contextual information embedded in the historical data.
Finally, in order to unveil the "black box" of artificial intelligence, we employed the SHapley Additive exPlanations (SHAP) method to interpret and analyze the prediction results of the self-attention GRU model with historical data. This provides insights into the significance and contributions of different input features on the pricing of American-style options.

<div style="display: flex; justify-content: space-around;">

<img src="https://pattishen1230.github.io/images/paper1-MLP.jpg" alt="Fig1" width="33%" />
<img src="https://pattishen1230.github.io/images/paper1-LSTM.jpg" alt="Fig2" width="33%" />
<img src="https://pattishen1230.github.io/images/paper1-self-attention.jpg" alt="Fig3" width="33%" />

</div>



## Contribution
- Construct four different machine learning models, including multilayer perceptron (MLP), long short-term memory (LSTM), self-attention LSTM, and self-attention gated recurrent unit (GRU) in comparison to the traditional binomial model.
- The results show that the self-attention GRU model with historical data outperforms other models due to its ability to capture complex temporal dependencies embedded in the historical data.
- Employ the SHapley Additive exPlanations (SHAP) method to interpret and analyze the prediction results of the self-attention GRU model with historical data, which provides insights that the current spot price, strike, and moneyness have the greatest impacts on predicting the current SPY call price.

## Citation
```
@article{shen2023american,
  title={American Option Pricing using Self-Attention GRU and Shapley Value Interpretation},
  author={Shen, Yanhui},
  journal={arXiv preprint arXiv:2310.12500},
  year={2023}
}
```
