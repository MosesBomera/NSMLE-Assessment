### Approach Summary for Loan Performance Prediction

**Objective:**
The primary goal is to predict loan performance, specifically whether a loan will be paid late, using historical loan application and repayment data. The training is set up in such a way that the prediction model is trained on past data and tested on future data to prevent data leakage and ensure accurate performance evaluation.

**Assumptions on Data:**
- The `paid_late` variable captures yield obtained, total owing at issue paid within the expected period from the loan i.e. that a loan cannot be paid on time without the yield and total owing amount being paid.
- There are potential errors in the values of `total_recovered_on_time` and `total_recovered_15_dpd`, as it is expected that if someone pays on time, they should have no payments after the due date.

**Target Variable:**
- The target variable for prediction is `paid_late`. From the EDA and problem description, this variable captures the general definitions of performance in one variable i.e. how much of the `total_owing_at_issue` was paid on time and after the due date, and if the yield was paid.
- The task is classification due to the categorical target variable.

**Data Points:**
- **Business Details:** Information about the business applying for the loan, such as employee count etc.
- **Loan Application Details:** Details about the loan application, including principal amount, total owing at issue, application number, loan number, and approval status.
- **Repayment History:** Historical data on loan repayments, including transaction types, amounts, and dates.

**Feature Engineering:**
1. **Historical Loan Features:**
   - For each loan application, extract historical details from previous loan applications considering a given business id.
   - Features include the number of previous applications, mean principal and owing amounts, number of late payments, and counts of different approval statuses.
   - The `loan_id` of the last approved loan is also included as a means to extract the performance of the last approved loan from the historical repayment features data.

2. **Repayment Features:**
   - Extract features from the repayment data, including the total amount paid, number of payments, maximum payment amount, duration of repayment, and whether the loan was paid in a single repayment.
   - Calculate counts and totals for different transaction types (e.g., `Deposit`, `Discount`).

3. **Training Data Preparation:**
   - Merge the historical loan features and repayment features based on the `loan_id` and `last_approved_loan_id`.
   - This ensures that each training instance contains only the variables that can be known at inference time, including loan application details, business details, loan behaviors before the current application, and repayment behavior before the current loan application.
   - For each current loan application, the corresponding `paid_late` value is concated to the training instance, this is the target.

**Preventing Data Leakage:**
- **Sorting and Grouping:** The data is sorted by `business_id`, `application_number`, and `loan_number` to ensure the temporal order of loan applications. This sorting ensures that the historical data used for feature extraction is always from the past relative to the current loan application.
- **Historical Feature Extraction:** By extracting features from previous loan applications and repayments, we ensure that the model is trained on historical data and tested on future data. This prevents data leakage by maintaining the temporal order of the data.
- **Train-Test Split:** `TimeSeriesSplit` from `sklearn.model_selection` is used to split the data into training and testing sets for cross validation. This ensures that the training set contains historical loan applications from the past and the test set contains future loan applications. The temporal order is maintained by sorting the data before applying `TimeSeriesSplit`.

**Training Process:**
- The data is sorted by `business_id`, `application_number`, and `loan_number` to maintain the temporal order within each business.
- Use `GridSearchCV` with `TimeSeriesSplit` as the cross-validator to search for the best model parameters while maintaining the temporal order of the data.
- Train the model on the training set and evaluate its performance on the test set to ensure accurate prediction of loan performance.

### Next Steps

**Framing the ML task to find a reason to lend a business:**
- Change the task to a regression task that predicts how much a business can manage to pay on time instead of if a loan will be paid late,
- This target can be calculated based on what the busines has historically managed to pay on time and within 15 days after the due date.

**Error Analysis:**
- Extensively analyse model errors to understand the causes and how to correct them to improve model performance.
- Devise strategies to handle the extreme class imbalance that exists in the `paid_late` target i.e (97% False to 3% True).

**Improve Feature Engineering:**
- Extract mean feature metrics from all previous loan repayments, not just the last approved loan.
- Consider only approved loan applications when extracting features for historical loan applications.

**Inference Pipeline Implementation:**
- Implement the inference pipeline for making predictions in a production setting.
- Pipeline will also be used when validating predictions on the provided test set.

**ML Workflow Managament Improvements:**
- Include an ML experimentation management module such as mlflow.

**Technical Improvements**
- Convert train_utils into a pip installable module.
- Add paramter, data validation using Pydantic.
- Add unit tests to train utilities.
