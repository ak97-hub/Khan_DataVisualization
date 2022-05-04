from scipy.sparse.construct import random
import streamlit as st
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.m_estimate import MEstimateEncoder
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from streamlit.legacy_caching.caching import cache
from streamlit.state.session_state import SessionState
import shap

header = st.container()
dataset = st.container()
encodings = st.container()
dcol1, dcol2 = st.columns(2)
selection = st.container()
scaling_data = st.container()
scol1, scol2 = st.columns(2)
stxt = st.container()
multicollinearity_cont = st.container()
modeling = st.container()
performance0= st.container()
performance1, performance2 = st.columns(2)
performance3 = st.container()
performance4 = st.container()
import streamlit.components.v1 as components
#mcc_text = st.container()


numeric_columns = [1, 2, 3, 4, 5, 6, 7, 8, 9]
ordinal_columns = [0]
cat_columns= [10]
data=pd.read_csv("brazilian-ecommerce-dataset")
data=data.iloc[0:10000,:]
target="price"


@st.cache
def encode_data(df, target, encode_selection, cat_cols):
    df = df.copy()
    if encode_selection == "Target encoder":
        te = TargetEncoder()
        data= te.fit_transform(df.iloc[:, cat_cols], df[target])
    elif encode_selection == "JamesStein":
        js = JamesSteinEncoder()
        data = js.fit_transform(df.iloc[:, cat_cols], df[target])
    df.iloc[:,cat_cols]=data
    return df.copy()

@st.cache
def scale_data(data, numeric_cols):
    data = data.copy()
    mm = StandardScaler()
    mmft= mm.fit_transform(data.iloc[:,numeric_cols])
    data.iloc[:,numeric_cols]= mmft
    return data.copy()

@st.cache
def multicollinearity_func(df):
    num_df = df.iloc[:,np.where(df.dtypes!=object)[0]]
    vif_info = pd.DataFrame()
    vif_info['VIF'] = [variance_inflation_factor(num_df.values, i) for i in
                       range(num_df.shape[1])]
    vif_info['Column'] = num_df.columns
    vif_info = vif_info.sort_values(by="VIF", ascending=False)
    return vif_info.copy()

def apply_model(data, target):
    X=data.drop(columns=target)
    y=data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=1234)
    gbr=GradientBoostingRegressor()
    gbr.fit(X_train, y_train)
    y_predict = gbr.predict(X_test)
    mse=mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mse)
    r2=gbr.score(X_test, y_test)
    scores = cross_val_score(gbr,
                             X_test,
                             y_test,
                             scoring="neg_mean_squared_error",
                             cv=5)
    rmse_scores = np.sqrt(-scores)
    results_dict = {"RMSE": [rmse],
                    "MSE": [mse],
                    "Mean RMSE Scores": [rmse_scores.mean()],
                    "Std RMSE Scores": [rmse_scores.std()],
                    "R2": [r2]
                    }
    res_df=pd.DataFrame.from_dict(data=results_dict)
    return X_train, X_test, y_test, y_predict, gbr, res_df
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


with header:
    st.title("Data Visualization of Gradient Boosting Regressor")
    st.subheader("Introduction")
    st.text(
    """
    This interactive application is to demonstrate the importance of
    visualization in the building of an Machine Learning(ML) model.
    Enjoy!
    """
    )

with dataset:
    dataset.header("Data")
    dataset.write(data.head(6))
    dataset.text(
    '''
    In this section you can change the encoder, and submit the change to see how
    each categorical column is effected by each specific encoder. Notice that in
    this section we only change categorical columns, that is simply because a
    machine learning algorithm cannot interpret what these categorical columns
    as strings, so we must convert these categorical columnns into numeric columns.
    However, given that these categorical columns now have numeric values, they
    could easily influence how the algorithm interprets connections. Therefore we
    must becareful how we encode these categorical columns. For example, a column
    that contains values bad, good, great can be denoted as an ordinal column
    because we can assign numeric values, such that bad = 0, good = 1,
    and great = 2. This works because we know that bad < good < great, and therefore
    0 < 1 < 2. Here the only ordinal column is "reviews".
    Each encoding available in the selectbox is just one of few, each using different
    techniques to overcome the problem of assigning numeric values without having an
    effect on how the model would otherwise interpret them and their relationships
    with other variables.

    Begin by Changing the encoder and see the effects below in the next section!
    More information:
    https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
    ''')
    st.markdown("***")
    st.header("Encoding Categorical Columns")
    with encodings:
        with selection:
            encoding_type = ["JamesStein", "Target encoder"]
            selection_encode = st.selectbox("Choose Encoder for categorical data",
            encoding_type)

        with dcol1:
            dcol1.header("Original Dataset")
            dcol1.write(data.iloc[:,[10,0,1, 2, 3, 4, 5, 6, 7, 8, 9]].head(6))
            size = "Size: 205 x 26"
            dcol1.text(size)

        with dcol2:
            dcol2.header("Encoded Dataset")
            data = encode_data(df=data,
             encode_selection=selection_encode,
             target=target,
             cat_cols=[10])
            dcol2.write(data.iloc[:,[10,0,1, 2, 3, 4, 5, 6, 7, 8, 9]].head(6))
            size = "Size: 205 x 26"
            dcol2.text(size)

        dataset.text(
        '''
        Here we can see that the categorical columns are converted into numbers.
        Also you are able to choose which method you would like to use to convert
        your data. Choose either the James Stein Encoder or the Target Encoder,
        and see how the change works on the categorical column
        "product_category_name"!
        '''
        )
    with scaling_data:
        scaling_data.markdown("***")
        scaling_data.header("Scaling the Data")
        scaling_data.text(
            """
            Now that our data is all encoded into numeric values, we need to move on to the next
            step. That is scaling our data. As you can see in the bottom left graph, this is a
            plot of the kernal density estimate(kde). The bottom left is the
            kde plot before scaling our data, and on the right of it is the
            kde plot of the scaled data.
            """)
        with scol1:
            fig, ax = plt.subplots()
            data.plot.kde(figsize=(5,5), grid=False, title="Before Scale",
            ax=ax)
            ax.legend(prop=dict(size=7))
            scol1.pyplot(fig)
        with scol2:
            data = scale_data(data, (numeric_columns+cat_columns))
            fig, ax = plt.subplots()
            data.plot.kde(figsize=(5,5), grid=False, title="After Scale", ax=ax)
            ax.legend(prop=dict(size=7))
            scol2.pyplot(fig)
        with stxt:
            stxt.text("***This was done using Standardized Scaling***")

with multicollinearity_cont:
    multicollinearity_cont.markdown("***")
    multicollinearity_cont.header("Multicollinearity")
    multicollinearity_cont.text(
        """
        Now that our data is all encoded into numeric values and scaled.
        We will move on to checking for high multicollinearity, in our data.
        """)
    vif_temp= multicollinearity_func(data)
    fig, ax = plt.subplots()
    pal= sns.color_palette("Set2", len(vif_temp))
    sns.barplot(data=vif_temp,x="VIF",y="Column",
     palette=pal[::-1], ax= ax)
    ax.set_title("VIF Bar Graph")
    multicollinearity_cont.pyplot(fig)
    multicollinearity_cont.text('''
    Here we can see that all data demonstrate healthy levels of
    multicollinearity. So we will not remove any
    ''')

with modeling:
    modeling.markdown("***")
    modeling.header("Modeling the Data")
    modeling.subheader('''
    Let's Guess Product Prices using GradientBoostingRegressor
    ''')
    modeling.text('''
    For this part we will be fitting a basic GradientBoostingRegressor model to
    see if we can guess the price of the product given the other variables.
    Check out below as we attempt to understand the performance of the model
    ''')
    modeling.markdown("***")
    modeling.subheader("Model Performance, Scores")
    X_train, X_test, y_test, y_predict, gbr, res_df = apply_model(data,
    target="price")
    shap.initjs()
    performance0.write(res_df)
    ft_import=pd.DataFrame(data={"Feature Importance":gbr.feature_importances_,
                   "Feature":data.drop(columns=target).columns})
    ft_import= ft_import.sort_values(by="Feature Importance", ascending=False)
    fig,ax = plt.subplots(figsize=(5,4.9))
    sns.scatterplot(y=y_predict,x=y_test, ax =ax)
    ax.set_ylabel("Predicted", fontsize = 12)
    ax.set_xlabel("Actual", fontsize = 12)
    ax.set_title("Predicted v. Actual")
    performance0.subheader("Model Performance, Plots")
    performance1.pyplot(fig)
    performance1.text('''
    From here we can see the plot of the
    predicted v.s. actual. This plot
    demonstrates some linear relationship.
    However, as shown in the table above
    our R2 score isn't greatest at about
    0.6. The model could definitely be
    improved, but that is outside the
    scope of this class.
    ''')
    fig, ax = plt.subplots(figsize=(5,8))
    sns.set(style="whitegrid", color_codes=True)
    pal=sns.color_palette("Greens_d", len(ft_import))
    sns.barplot(data=ft_import, x="Feature Importance",
            y="Feature",
            palette=pal[::-1],
            ax=ax)
    performance2.pyplot(fig)
    performance2.text('''
    Here we are able to see the importance
    of each feature, with product weight
    being the best feature for guessing
    the price of the product. Although,
    it is important to note that these
    feautures work in conjunction, and so
    the feauture importance is tied to it's
    interactions with all other features.
    ''')
    explainer = shap.Explainer(gbr)
    expected_value = explainer.expected_value
    shap_array = explainer.shap_values(X_train)
    #st_shap(shap.force_plot(expected_value,
    #shap_array[0,:], X_train.iloc[0,:]))
    with performance3:
        performance3.subheader("Shap Decision Plot, 50 Observations")
        fig, ax = plt.subplots(figsize=(5,8))
        ax = shap.decision_plot(expected_value, shap_array[50:100],
                       feature_names=list(data.drop(columns=target).columns))
        performance3.pyplot(fig)
        performance3.text('''
        Here we are able to see a parellel plot of our shap values. This helps,
        us further understand how each feauture effects observations in the
        dataset. From here we can get insights to which features are more
        influential. Although, this was limited to only 50 observations, because
        too many observations can overcrowd the visual. This could be a
        limitatition, given we are only able to observe small parts of the data
        at a time.
        ''')
    with performance4:
        performance4.subheader("Beeswarm Shap Plot")
        fig, ax = plt.subplots(figsize=(5,8))
        shap_array = explainer(X_train)
        ax = shap.plots.beeswarm(shap_array)
        performance4.pyplot(fig)
        performance4.text('''
        Again, this plot is no different than the previous ones in that it is
        also just another way to visualize the feature importance according to
        the SHAP values.
        ''')
