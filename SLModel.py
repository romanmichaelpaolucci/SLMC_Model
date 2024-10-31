import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Define the column headers (based on previous discussions)
column_headers = [
    "Reference Pool ID", "Loan Identifier", "Monthly Reporting Period", "Channel",
    "Seller Name", "Servicer Name", "Master Servicer", "Original Interest Rate",
    "Current Interest Rate", "Original UPB", "UPB at Issuance", "Current Actual UPB",
    "Original Loan Term", "Origination Date", "First Payment Date", "Loan Age",
    "Remaining Months to Legal Maturity", "Remaining Months To Maturity", "Maturity Date",
    "Original Loan to Value Ratio (LTV)", "Original Combined Loan to Value Ratio (CLTV)",
    "Number of Borrowers", "Debt-To-Income (DTI)", "Borrower Credit Score at Origination",
    "Co-Borrower Credit Score at Origination", "First Time Home Buyer Indicator", "Loan Purpose",
    "Property Type", "Number of Units", "Occupancy Status", "Property State",
    "Metropolitan Statistical Area (MSA)", "Zip Code Short", "Mortgage Insurance Percentage",
    "Amortization Type", "Prepayment Penalty Indicator", "Interest Only Loan Indicator",
    "Interest Only First Principal And Interest Payment Date", "Months to Amortization",
    "Current Loan Delinquency Status", "Loan Payment History", "Modification Flag",
    "Mortgage Insurance Cancellation Indicator", "Zero Balance Code", "Zero Balance Effective Date",
    "UPB at the Time of Removal", "Repurchase Date", "Scheduled Principal Current",
    "Total Principal Current", "Unscheduled Principal Current", "Last Paid Installment Date",
    "Foreclosure Date", "Disposition Date", "Foreclosure Costs", 
    "Property Preservation and Repair Costs", "Asset Recovery Costs",
    "Miscellaneous Holding Expenses and Credits", "Associated Taxes for Holding Property",
    "Net Sales Proceeds", "Credit Enhancement Proceeds", "Repurchase Make Whole Proceeds",
    "Other Foreclosure Proceeds", "Principal Forgiveness Amount", "Modification-Related Non-Interest Bearing UPB",
    "Original List Start Date", "Original List Price", "Current List Start Date", "Current List Price",
    "Borrower Credit Score At Issuance", "Co-Borrower Credit Score At Issuance", "Borrower Credit Score Current",
    "Co-Borrower Credit Score Current", "Mortgage Insurance Type", "Servicing Activity Indicator",
    "Current Period Modification Loss Amount", "Cumulative Modification Loss Amount",
    "Current Period Credit Event Net Gain or Loss", "Cumulative Credit Event Net Gain or Loss",
    "Special Eligibility Program", "Foreclosure Principal Write-off Amount",
    "Relocation Mortgage Indicator", "Zero Balance Code Change Date", "Loan Holdback Indicator",
    "Loan Holdback Effective Date", "Delinquent Accrued Interest", "High Balance Loan Indicator",
    "Property Valuation Method", "ARM Product Type", "Initial Fixed-Rate Period",
    "Interest Rate Adjustment Frequency", "Next Interest Rate Adjustment Date",
    "Next Payment Change Date", "Index", "ARM Initial Fixed-Rate Period ≤ 5 YR Indicator",
    "ARM Cap Structure", "Initial Interest Rate Cap Up Percent",
    "Periodic Interest Rate Cap Up Percent", "Lifetime Interest Rate Cap Up Percent",
    "Mortgage Margin", "ARM Plan Number", "ARM Balloon Indicator", "Borrower Assistance Plan",
    "High Loan to Value (HLTV) Refinance Option Indicator", "Deal Name",
    "Repurchase Make Whole Proceeds Flag", "Alternative Delinquency Resolution Count",
    "Alternative Delinquency Resolution", "Total Deferral Amount", "Payment Deferral Modification Event Indicator",
    "Interest Bearing UPB"
]

# To display the entire app not conditional on the matrix existing
# TODO: Message in transition matrix/stats/calculator if data has not been uploaded
# TODO: Fix this preprocessing so the entire app shows not just when there is data...?
st.session_state.transition_matrix = pd.DataFrame(columns=['Current', '30 Days Delinquent', '60 Days Delinquent', '90+ Days Delinquent'])
st.session_state.unique_loans = 0
st.session_state.total_loan_value = 0
st.session_state.avg_loan_balance = 0
st.session_state.delinquency_rate = 0
st.session_state.expected_loss = 0

# Function to categorize delinquency status
def categorize_delinquency_status(status):
    """Bucket delinquency status into predefined states."""
    if status == 0:
        return 'Current'
    elif status == 1:
        return '30 Days Delinquent'
    elif status == 2:
        return '60 Days Delinquent'
    elif status >= 3:
        return '90+ Days Delinquent'
    #elif status == -1:
    #    return 'Paid Off'
    else:
        return 'Unknown'

# Title of the application
st.header("Markov Chain Single-Family Model", divider=True)

st.write("This app enables the user to develop a markov chain model for single-family loans available from Fannie Mae. Linked on the sidebar is the original data source along with an example of chunks to upload to the application (remember chunk uploads must be contiguous for state transitions!).")

st.write("**This app is hosted using Streamlit, large data may not be processed on this server and is best handled on a local instance of this app. Please clone this repository to run the app locally on your machine: https://github.com/romanmichaelpaolucci/SLMC_Model**")

# Sidebar Feature Toggles
st.sidebar.title("Model Settings")
lookback_period = st.sidebar.slider("Lookback Period (Months)", min_value=1, max_value=12, value=3)
lgd = st.sidebar.number_input("Loss Given Default (LGD)", min_value=0.0, max_value=1.0, value=0.4, step=0.01)

st.sidebar.write("Data Source: https://datadynamics.fanniemae.com/data-dynamics/#/downloadLoanData/Single-Family")
st.sidebar.write("Example of Data Format to Upload: https://drive.google.com/drive/folders/1E0qwFf4Cg5Xw69ybYzorTticVxBHkaWO?usp=sharing")

# Sidebar Author Section
st.sidebar.title("About the Author")
author_image = Image.open("roman_image.png")  # Make sure this image is in the same directory
st.sidebar.image(author_image, caption="Incoming Summer Analyst @ KBRA", use_column_width=True)

st.sidebar.write("Author: Roman Paolucci")
st.sidebar.write("Graduate Student @ Columbia")

# Social Media Links with Icons
st.sidebar.markdown(
    """
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rmp99/)
    [![GitHub](https://img.shields.io/badge/GitHub-333?style=for-the-badge&logo=github&logoColor=white)](https://github.com/romanmichaelpaolucci)
    """
)

st.header("Statistical Parameter Estimation", divider=True)
# Methodology Section with LaTeX rendering
with st.expander("Markov Chain Parameter Estimation"):
    st.write(r"""
    ### Transition Matrix Estimation Methodology

    Given a set of possible states $S = \{ \text{Current}, \text{30 Days Delinquent}, \text{60 Days Delinquent}, \text{90+ Days Delinquent}, \text{Paid Off} \}$,
    the goal is to estimate a transition probability matrix $P$, where each entry $P_{ij}$ represents the probability of transitioning from state $i$ to state $j$.

    #### Maximum Likelihood Estimation (MLE)
    
    Using observed data, the transition probabilities can be estimated using Maximum Likelihood Estimation (MLE). Suppose $N_{ij}$ denotes the number of observed transitions from state $i$ to state $j$ across all loan sequences. The MLE for the transition probability $P_{ij}$ is given by:
    
    $\hat{P}_{ij} = \frac{N_{ij}}{\sum_{k} N_{ik}}$

    where $\sum_{k} N_{ik}$ is the total number of transitions observed from state $i$ to any other state. This ensures that each row in the transition matrix sums to 1.

    #### Step-by-Step Process

    1. **Categorize Loan States**: Each loan's delinquency status is categorized into discrete states such as *Current*, *30 Days Delinquent*, etc.
    2. **Calculate Transitions**: For each loan, calculate transitions between these states by observing consecutive periods.
    3. **Estimate Transition Matrix**: Using MLE, we compute the probability of moving from one state to another as shown above.
    """)

st.write("More on Parameter Estimation: https://towardsdatascience.com/maximum-likelihood-estimation-4a1a866dfa70")

st.header("Lookback Period for Parameter Estimation", divider=True)
# Explanation of Lookback Period Selection
with st.expander("Lookback Period Selection Explanation"):
    st.write("""
    The lookback period determines the number of consecutive months used to estimate transition probabilities. 
    Here’s why different windows might be preferred:
    
    - **3-6 Months**: Often chosen for shorter-term risk assessment, capturing recent borrower behavior and adapting to economic shifts.
    - **12 Months**: Common in annual risk assessments, this window smooths seasonal patterns and is often used in regulatory models.
    - **18+ Months**: Longer periods may provide stable trends in risk but can dilute recent changes, so they’re usually used in strategic, longer-term models.
    """)

# Cached function to load and process data
@st.cache_data
def load_and_process_data(uploaded_files, lgd):
    data_frames = []
    for file in uploaded_files:
        df_chunk = pd.read_csv(file, sep='|', header=None, names=column_headers)
        data_frames.append(df_chunk)
    df = pd.concat(data_frames, ignore_index=True)

    # Convert columns to numeric types
    df["Current Loan Delinquency Status"] = pd.to_numeric(df["Current Loan Delinquency Status"], errors='coerce')
    df["Current Actual UPB"] = pd.to_numeric(df["Current Actual UPB"], errors='coerce')

    # Step 1: Apply categorization
    df['State'] = df['Current Loan Delinquency Status'].apply(categorize_delinquency_status)
    df = df.sort_values(['Loan Identifier', 'Monthly Reporting Period'])
    df['Previous State'] = df.groupby('Loan Identifier')['State'].shift(1)
    df_transitions = df.dropna(subset=['Previous State'])

    # Step 2: Calculate transition counts and matrix
    all_states = ['Current', '30 Days Delinquent', '60 Days Delinquent', '90+ Days Delinquent']#, 'Paid Off']
    transition_counts = df_transitions.groupby(['Previous State', 'State']).size().unstack(fill_value=0)
    transition_counts = transition_counts.reindex(index=all_states, columns=all_states, fill_value=0)
    transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0)
    transition_matrix.loc['Current', '60 Days Delinquent'] = 0
    transition_matrix.loc['Current', '90+ Days Delinquent'] = 0
    transition_matrix.loc['30 Days Delinquent', '90+ Days Delinquent'] = 0
    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

    # High-level statistics
    unique_loans = df["Loan Identifier"].nunique()
    total_loan_value = df.groupby("Loan Identifier")["Current Actual UPB"].max().sum()
    avg_loan_balance = df["Current Actual UPB"].mean()
    delinquency_rate = (df["Current Loan Delinquency Status"] > 0).mean()
    expected_loss = delinquency_rate * total_loan_value * lgd

    return transition_matrix, unique_loans, total_loan_value, avg_loan_balance, delinquency_rate, expected_loss

st.header("Single-Family Data Upload", divider=True)
# File uploader
uploaded_files = st.file_uploader("Upload your loan data CSV files (in chunks)", type="csv", accept_multiple_files=True)

# Process files if uploaded
if uploaded_files:
    transition_matrix, unique_loans, total_loan_value, avg_loan_balance, delinquency_rate, expected_loss = load_and_process_data(uploaded_files, lgd)

    # Store processed data in session state
    st.session_state.transition_matrix = transition_matrix
    st.session_state.unique_loans = unique_loans
    st.session_state.total_loan_value = total_loan_value
    st.session_state.avg_loan_balance = avg_loan_balance
    st.session_state.delinquency_rate = delinquency_rate
    st.session_state.expected_loss = expected_loss

    # Display the transition matrix
    st.write("Transition Probability Matrix with Logical Constraints")
    st.dataframe(st.session_state.transition_matrix)

# Display High-Level Statistics with hoverable Info Icons
st.header("Portfolio Summary Statistics", divider=True)
st.write(f"- **Unique Loans**: {st.session_state.unique_loans}")
st.write(f"- **Total Loan Value**: ${st.session_state.total_loan_value:,.2f}")
st.write(f"- **Average Loan Balance**: ${st.session_state.avg_loan_balance:,.2f}")
st.write(f"- **Delinquency Rate**: {st.session_state.delinquency_rate:.2%}")
st.write(f"#### Expected Loss based on LGD: ${st.session_state.expected_loss:,.2f}")

st.header("Forward Looking Probability & Expectation", divider=True)
# Transition Probability Calculator using Chapman-Kolmogorov equation
if 'transition_matrix' in st.session_state:
    with st.expander("Chapman-Kolmogorov Equation Explanation"):
        st.write(r"""
        The Chapman-Kolmogorov equation allows us to compute the probability of transitioning from one state to another
        over multiple steps. Given a transition matrix $ P $ and a number of steps $ n $, the probability of transitioning
        from state $ i $ to state $ j $ in $ n $ steps is given by $ P^n_{ij} $.

        This can be computed by raising the transition matrix $ P $ to the power $ n $.

        For a full proof see:
        """)
        st.video("https://www.youtube.com/watch?v=L3FqYBDw9fE")
    
    
    st.write("### Transition Probability Calculator")
    start_state = st.selectbox("Select Start State", list(st.session_state.transition_matrix.columns), key="start_state")
    end_state = st.selectbox("Select End State", list(st.session_state.transition_matrix.columns), key="end_state")
    num_steps = st.number_input("Number of Months", min_value=1, max_value=200, value=1, key="num_steps")

    if st.session_state.transition_matrix.size == 0:
        st.write("Please Upload Data in Appropriate Format (See Sidebar for Example!)")
    else:
        # Calculate the n-step transition probability using Chapman-Kolmogorov without re-running the data pipeline
        try:
            # Raise the transition matrix to the power of 'num_steps' to get multi-step transition probabilities
            n_step_transition_matrix = np.linalg.matrix_power(st.session_state.transition_matrix.values, num_steps)
            probability = n_step_transition_matrix[st.session_state.transition_matrix.index.get_loc(start_state), 
                                                   st.session_state.transition_matrix.columns.get_loc(end_state)]
            st.write(f"Probability of transitioning from **{start_state}** to **{end_state}** in **{num_steps}** steps: {probability:.4f}")
        except Exception as e:
            st.write("Error calculating transition probability:", e)

# Absorbing State Modeling for 90+ Days Delinquent
if 'transition_matrix' in st.session_state:
    
     # Modify transition matrix to treat "90+ Days Delinquent" as an absorbing state
    absorbing_matrix = st.session_state.transition_matrix.copy()
    absorbing_matrix.loc['90+ Days Delinquent'] = [0, 0, 0, 1] # extra 0 for paid off state, 0]  # Make 90+ Days Delinquent an absorbing state

    # Display the adjusted transition matrix with absorbing state
    st.header("90+ Days Delinquent as Absorbing State", divider=True)
    with st.expander("Absorbing State Modeling for 90+ Days Delinquent"):
        st.write(r"""
        In this model, we consider the "90+ Days Delinquent" state as an absorbing state, which means that once a loan
        enters this state, it is assumed to remain there indefinitely. This assumption aligns with treating loans in
        severe delinquency as defaulted.

        The implications are significant: if a loan enters this state, it is unlikely to transition back to other states.
        This approach can be used to calculate the probability of eventual default by observing the transition probabilities
        into this absorbing state.
        """)
    st.dataframe(absorbing_matrix)

    # Transition Probability Calculator for the Absorbing State Matrix
    st.write("### Absorbing State Transition Probability Calculator")
    start_state_absorbing = st.selectbox("Select Start State (Absorbing Matrix)", list(absorbing_matrix.columns), key="start_state_absorbing")
    end_state_absorbing = st.selectbox("Select End State (Absorbing Matrix)", list(absorbing_matrix.columns), key="end_state_absorbing")
    num_steps_absorbing = st.number_input("Number of Months (Absorbing Matrix)", min_value=1, max_value=200, value=1, key="num_steps_absorbing")

    if st.session_state.transition_matrix.size == 0:
        st.write("Please Upload Data in Appropriate Format (See Sidebar for Example!)")
    else:
        # Calculate the n-step transition probability for the absorbing state matrix
        try:
            # Raise the modified transition matrix to the power of 'num_steps_absorbing'
            n_step_absorbing_matrix = np.linalg.matrix_power(absorbing_matrix.to_numpy(), num_steps_absorbing)
            probability_absorbing = n_step_absorbing_matrix[absorbing_matrix.index.get_loc(start_state_absorbing),
                                                            absorbing_matrix.columns.get_loc(end_state_absorbing)]
            st.write(f"Probability of transitioning from **{start_state_absorbing}** to **{end_state_absorbing}** in **{num_steps_absorbing}** steps: {probability_absorbing:.4f}")
        except Exception as e:
            st.write("Error calculating transition probability in absorbing matrix:", e)