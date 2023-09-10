# Home Assignment - Test Output Prediction

---
## **Note on Phrasing**

While I pride myself on my English proficiency, some of the eloquence in phrasing was given a boost with the help of ChatGPT. It's like having a friendly grammar enthusiast looking over your shoulder! 😉

---




## **Disclaimer**

This documentation aims to capture the essence and key steps of the work done. While I've made a concerted effort to provide a thorough overview, certain intricacies, minor steps, or specific decision rationales might not be exhaustively detailed here.

It's worth noting that the choices and methods undertaken were based on my interpretation and understanding of the problem at hand. While I stand by the decisions made in this context, in real-world scenarios, I always advocate for collaborative brainstorming sessions. Engaging product teams, fellow data scientists, and other stakeholders ensures comprehensive insights and a holistic decision-making process.

What's presented here is a glimpse into the methodology, and while I've strived for clarity and accuracy, collaboration remains a cornerstone of successful and well-rounded data-driven projects.

---

**Note: To be able to effectivly use the links in this document, clone the repo and read this from your local jupyter notebook/lab server**


## **Executive Summary**

#### **Objective**: 
The aim was to predict the pass/fail status of a high-cost test, TLJYWBE, using machine learning techniques to ultimately reduce the frequency with which this expensive test needs to be run.

#### **Problem Context**:
- The test, TLJYWBE, outputs continuous numbers. A pass/fail threshold is set at $10^{-5}$.
- A result equal to or greater than this threshold indicates a test failure; otherwise, it's a pass.
- The dataset's ratio of passed to failed tests is approximately 1:11347, signifying a highly imbalanced scenario.
- Given the significant consequences of an undetected failed test, the primary focus is on achieving 100% recall. At the same time, we aim to maximize precision, which in turn would reduce the number of test runs.
- The comprehensive dataset comprises around 720k samples, with each undergoing a subset of 880 tests.

#### **Data Preprocessing**:
- Removed tests that were identical, yielding only one output, or were highly correlated (correlation > 0.9).
- Rectified numerous irregular values present in the data.

#### **Modeling**:
- Adopted tree ensemble classifications as the primary modeling approach.
- Utilized a Random Forest (RF) classifier for feature identification on a downsampled dataset.
- Constructed a pipeline and executed a grid search for RF, BalancedRF, and EasyEnsemble classifiers on two datasets: one with 760 features and another with 367 features.
- Resampled the 367-feature dataset using both upsampling (SMOTE) and downsampling (TomekLink) techniques, followed by grid search on the resampled dataset.

#### **Key Results**:
- Some tests were discovered to be identical or 100% correlated. Those can be avoided to save costs. 
- After rigorous preprocessing and the application of an appropriate modeling technique, the ML model successfully **reduced the number of required TLJYWBE tests by 85%**, using 760 tests as input features.


## **The buisness case & Metric of use**

As we've touched on, the failed-to-passed ratio of the test stands strikingly at 1:11347. When you look at roughly 720k tests and see that only 64 have failed, it naturally begs the question: Do we genuinely need this test? Would it make sense to just predict a "pass" for every sample? Sure, we'd make 64 errors, but we'd also sidestep 720k test runs. However, because we're aiming to predict the outcome of this test, the answer, at least in my mind, has to be a resounding "No". Missing even one failure of this test might be far more costly (perhaps by a long shot) than the savings from avoiding 11347 runs of this pricey test.

From this perspective, I felt that the metric we should be focusing on is **precision at 100% recall**. This metric is the last point on a precision-recall curve. It gives us measure of the proportion of true positives we've got in the samples we predicted as positive when we're ensuring a full 100% recall (by tweaking the threshold right). If we manage to nail a 100% in precision at 100% recall, then everything we're flagging as positive genuinely is. In that case, we could potentially avoid all the 720k tests, cutting down test costs by a whole 100%. 

Perfect scores are what we aim for, but one needs to be realistic. To put things into perspective, a 1% precision at 100% recall means that for every correct positive prediction, we'd get 99 false alarms. But even then, we'd only need to conduct 64 times 99 tests instead of 720k. That still gives us a 99% cut in testing costs.

Since sklearn does not support such a metric, I've created my own metric and scorer, which can be found [here](src/custom_metrics.py).

## **Preprocessing**

You can view the detailed preprocessing notebook [here](1.%20Data%20cleansing.ipynb).

**Key steps I undertook:**
- Checked for and removed any duplicate rows.
- Rectified a variety of irregular values in the data, including `np.inf`, string representations of NaNs, infinity symbols, and booleans represented as strings.
- Excluded 16k entries where the target column had null values.
- Addressed columns that were [exact duplicates](1.%20Data%20cleansing.ipynb#Handling-duplicated-columns).
- Retained columns with only one unique non-null value. I included these columns in the larger 760-feature dataset because it was unclear if the presence of null values was intentional or an oversight.
- Completed a [missing values analysis](https://github.com/edoson/nvidia-home-assigment/blob/main/1.%20Data%20cleansing.ipynb#Correlation-Analysis).
- Conducted a [correlation analysis](1.%20Data%20cleansing.ipynb#Correlations) and removed 442 features that:
  - Had less than 50% non-null values.
  - Were highly correlated (> 0.9) with another column. The primary (source) column was retained.
  
**Outcomes:**
- Produced a dataset with 760 features, which includes both correlated and single-valued columns.
- Developed a refined dataset with 367 features, excluding the correlated and single-valued columns.
- While I haven't further pruned the dataset, there's potential for further reduction. Given that a correlation score of 0.9 is quite high, I believe more columns could be eliminated without significantly impacting performance.


## **Modeling**

**Disclaimer:** I explored several approaches during this project. Only the most successful ones remain documented in the repo, as I iterated on the same notebooks.

### **Modeling Steps**:
1. Developed a **pipeline** to process int, float, and categorical features:
   - Numerical missing values were imputed with medians.
   - Categorical missing values were labeled as 'missing'.
   - Numerical features were standardized.
   - Categorical features were one-hot encoded.
   
2. Employed various classifiers including:
   - `RandomForest` with class weights (from `sklearn`).
   - `BalancedRandomForestClassifier`, `EasyEnsembleClassifier`, and `RUSBoostClassifier` (from `imblearn`).
   
3. Conducted a **grid search** for optimal hyperparameters.

4. Assessed the best model's performance using the previously discussed `precision@100%recall` scorer.

5. Calculated the cost reduction based on the achieved precision score.

6. Applied this technique to both the 367-feature and 760-feature datasets.

### **Key Results**:
- Achieved an **85% cost reduction** using the [EasyEnsembleClassifier with 100 estimators](4.%20Modeling__760_features.ipynb#EasyEnsembleClassifier) on the 760-feature dataset.
- Attained a **75% cost reduction** with the [BalancedRandomForestClassifier](3.%20Modeling__376_features.ipynb#Test-model) on the 367-feature dataset.

### **Other Explorations**:
- Experimented with [reducing the feature space](2.%20Feature%20importance.ipynb). This involved a grid search with logistic regression and random forest classifiers on a downsampled majority class and assessing feature importance. However, due to non-overlapping features between the two and time constraints, I decided to forgo this approach.