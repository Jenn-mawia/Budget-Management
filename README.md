# Budget Management System

A data science project that analyzes M-Pesa and bank transaction data to identify spending patterns using KMeans clustering and incorporates Generative AI for deeper financial insights.

## Project Overview

This project helps users understand their spending behaviors by:
- Analyzing transaction data from M-Pesa and bank statements
- Identifying spending patterns using unsupervised machine learning (KMeans clustering)
- Providing AI-powered insights and recommendations for better budget management
- Visualizing spending trends and categories

## A. Setup Instructions

### Required Tools and Environment

- **Python**: Version 3.8 or higher
- **Jupyter Notebook**: For interactive analysis
- **Virtual Environment**: Recommended for dependency management
- **Git**: For version control

### Environment Setup

#### Option 1: Using virtualenv
```bash
# Create virtual environment
python -m venv budget-management-env

# Activate virtual environment
# On Windows:
budget-management-env\Scripts\activate
# On macOS/Linux:
source budget-management-env/bin/activate

# Install Jupyter
pip install jupyter
```

### Installation of Dependencies

Install required packages using the requirements file:

```bash
pip install -r requirements.txt
```

### Dataset Requirements

#### Data Sources
- **M-Pesa transaction data**: CSV format with transaction details   
- **Bank statement data**: CSV format with account transactions
  - Download your M-Pesa statement by dialing *234# on your Safaricom line.
  - Convert your statement to .xlsx format on this website[https://www.ilovepdf.com/]

#### Expected Data Format
Your transaction data should contain the following columns:
- `Date`: Transaction date (YYYY-MM-DD format)
- `Amount`: Transaction amount (numeric)
- `Description`: Transaction description/details
- `Category`: Transaction category (optional)
- `Transaction_Type`: Debit/Credit indicator

#### Sample Data Structure
```
Date,Amount,Description,Transaction_Type
2024-01-15,1500.00,Grocery Shopping,Debit
2024-01-16,50000.00,Salary,Credit
2024-01-17,800.00,Fuel,Debit
```

### Folder Structure
Budget-Management/
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── model.ipynb                 # Main analysis notebook 
├── mpesa_data.csv             # M-Pesa transaction dataset
├── mpesa.py                   # M-Pesa data processing functions -> Main python script
├── converter.py               # Data conversion utilities 
└── time_category_insights.py  # Time-based analysis and insights


### Environment Variables Setup

Create a `.env` file for API keys:
```bash
# .env file
OPENAI_API_KEY=your_openai_api_key_here
```

## B. How to Run the Project

### Step 1: Data Preparation
1. Place your transaction data in the root/main folder as shown above
2. Ensure data follows the expected format mentioned above and also has the specified columns

### Step 2: Running the Analysis

#### Option 1: Run the Notebook(model.ipynb) Sequentially for interactive outputs
```bash
# Start Jupyter Notebook
jupyter notebook

# Run notebook in order from the first cell:
# 1. model.ipynb
```

#### Option 2: Run as Python Scripts
```bash
# Run the complete pipeline on your local machine
streamlit run mpesa.py
```

### Step 3: Cell Execution Sequence

**In Jupyter Notebooks:**
1. **Importing Libraries**: Importing the necessary libraries
2. **Data Exploration**: Load and examine transaction data
3. **Data Cleaning and Preprocessing**: Clean and prepare data for analysis
4. **Feature Engineering**: Create new features that will help with the Clustering model
5. **Clustering Analysis/Modeling**: Apply KMeans to identify spending patterns
6. **Cluster Summaries**: Visualization of the different clusters' data
7. **AI Insights**: Generate recommendations using GenAI

### Expected Output

**Console Output:**
```
![Screenshot 2025-07-09 at 07 57 58](https://github.com/user-attachments/assets/03e8ddcd-c245-429f-ad97-91368ceace0c)
```

**Sample Clustering Results:**
- **Cluster 0**: Essential expenses (groceries, utilities) - 35% of spending
- **Cluster 1**: Transportation costs - 20% of spending  
- **Cluster 2**: Entertainment and dining - 25% of spending
- **Cluster 3**: Large purchases/investments - 10% of spending
- **Cluster 4**: Miscellaneous small transactions - 10% of spending
  
The number of clusters achieved entirely depends on your data

## C. Key Functions Documentation

### 1. `categorize_details()` Function

**Location**: Main script/notebook

**Purpose**: This function is essential for converting raw M-Pesa transaction descriptions into meaningful spending categories. It uses pattern matching to automatically classify transactions, enabling spending pattern analysis and budgeting insights.

**Function Signature**:
```python
def categorize_details(details):
    """
    Categorizes M-Pesa transaction details into spending categories
    
    Parameters:
    -----------
    details : str
        Raw transaction description from M-Pesa/bank statement
        
    Returns:
    --------
    str
        Categorized spending category
    """
```

**What it does**:
- **Pattern matching**: Uses keyword detection to identify transaction types
- **Spending categorization**: Groups transactions into 12+ meaningful categories
- **Kenyan context**: Tailored for local businesses and service providers
- **Fallback handling**: Returns "Other" for unrecognized patterns

**Input Format**:
```python
# Example transaction details:
details_examples = [
    "Paid to NAIVAS SUPERMARKET - KIMATHI STREET",
    "Airtime purchase for 0722123456",
    "Paid to KPLC PREPAID for 123456789",
    "Withdraw from Agent 7629905 - JOHN DOE"
]
```

**Output Categories**:
```python
# Supported spending categories:
categories = {
    'Airtime': ['airtime', 'safaricom', 'airtel', 'bundles'],
    'Power': ['kplc'],
    'Rent Payment': ['55478'],  # Specific agent code
    'Shopping': ['naivas', 'tuskys', 'carrefour', 'jumia', 'supermarket'],
    'Restaurant': ['java', 'hotel', 'restaurant', 'cafe'],
    'Savings': ['sacco', 'old mutual', 'mmf'],
    'Water': ['alpha-water', 'water'],
    'Beauty': ['beauty', 'nimo naturals', 'bandari'],
    'Butchery': ['butchery', 'meat', 'butcher'],
    'People Transfer': ['customer transfer'],
    'Withdrawals': ['withdraw'],
    'Transaction Charge': ['charge'],
    'Pay Bill': ['pay bill'],
    'Other': 'Default for unmatched patterns'
}
```

**Edge Cases & Assumptions**:
- Converts input to lowercase for case-insensitive matching
- Handles None/NaN values by converting to string
- Uses keyword lists for flexible pattern matching
- Prioritizes specific matches over general ones
- Assumes transaction descriptions contain recognizable keywords
- Returns "Other" as safe fallback for unknown patterns

**Usage Example**:
```python
# Categorization examples:
categorize_details("Paid to NAIVAS SUPERMARKET")  # Returns: "Shopping"
categorize_details("KPLC PREPAID")                # Returns: "Power"
categorize_details("Airtime for 0722123456")      # Returns: "Airtime"
categorize_details("Unknown merchant")            # Returns: "Other"
```



### 2. `load_data()` Function

**Location**: Main script/notebook

**Purpose**: This function is the core data loading and preprocessing pipeline for M-Pesa transaction data. It handles both uploaded XLSX files and the default CSV dataset, performing essential data cleaning and feature engineering for spending analysis.

**Function Signature**:
```python
@st.cache_data
def load_data(xlsx_file=None):
    """
    Loads and preprocesses M-Pesa transaction data from XLSX or CSV sources
    
    Parameters:
    -----------
    xlsx_file : file-like object, optional
        Uploaded XLSX file containing M-Pesa transaction data
        If None, defaults to 'mpesa_data.csv'
        
    Returns:
    --------
    pandas.DataFrame or None
        Processed dataframe ready for analysis, or None if loading fails
    """
```

**What it does**:
- **Multi-source data loading**: Handles both XLSX uploads and CSV fallback
- **Column validation**: Ensures required M-Pesa columns are present
- **Data cleaning**: Fills missing values and filters transaction types
- **Feature engineering**: Creates categorical, temporal, and amount features
  - Creates time-based features (day of week, month, hour)
  - Extracts spending categories from transaction descriptions
- **Transaction filtering**: Focuses on withdrawal transactions (Type=0)


**Input Format**:
```python
# Expected XLSX/CSV columns:
required_columns = [
    'Receipt No.',      # Transaction identifier
    'Completion Time',  # Transaction timestamp
    'Details',          # Transaction description
    'Transaction Status', # Success/failure status
    'Paid In',          # Money received
    'Withdrawn',        # Money sent/withdrawn
    'Balance'           # Account balance after transaction
]
```

**Output Format**:
```python
# Processed DataFrame with additional features
processed_df = pd.DataFrame({
    'Receipt No.': ['ABC123', 'DEF456'],
    'Completion Time': ['2024-01-15 14:30:00', '2024-01-16 09:15:00'],
    'Details': ['Paid to NAIVAS SUPERMARKET', 'Airtime purchase'],
    'Paid In': [0.0, 0.0],
    'Withdrawn': [1500.0, 100.0],
    'Balance': [45000.0, 43500.0],
    'Type': [0, 0],                    # 0=withdrawal, 1=payment received
    'Category': ['Shopping', 'Airtime'], # Categorized spending
    'Hour': [14, 9],                   # Hour of transaction
    'DayOfWeek': ['Monday', 'Tuesday'], # Day name
    'Month': [1, 1],                   # Month number
    'Amount': [-1500.0, -100.0]        # Calculated transaction amount
})
```
**Edge Cases & Assumptions**:
- Handles missing 'Details' by filling with 'Other'
- Fills missing 'Paid In' and 'Withdrawn' values with 0
- Assumes XLSX files may contain multiple sheets
-  Assumes transaction amounts are positive numbers
-  Assumes date format is parseable by pandas
- Filters to only analyze withdrawal transactions -> the focus is on spending patterns, so no need to look at "PAID IN" transactions
- Requires exact column name matches for XLSX processing
  

### 2. `perform_clustering_analysis()` Function

**Location**: `src/clustering.py`

**Purpose**: This is the core machine learning function that applies KMeans clustering to identify distinct spending patterns in the transaction data.

**Function Signature**:
```python
def perform_clustering_analysis(df, features=['Amount_Normalized', 'Day_of_Week', 'Month'], 
                               n_clusters=None, random_state=42):
    """
    Performs KMeans clustering on transaction data to identify spending patterns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed transaction data
    features : list, default=['Amount_Normalized', 'Day_of_Week', 'Month']
        Features to use for clustering
    n_clusters : int, optional
        Number of clusters. If None, optimal number is determined using elbow method
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (clustered_dataframe, kmeans_model, cluster_summary)
    """
```

**What it does**:
- Determines optimal number of clusters using the elbow method
- Applies KMeans clustering algorithm
- Assigns cluster labels to each transaction
- Generates cluster summary statistics
- Calculates cluster centroids and characteristics
- Validates clustering quality using silhouette score

**Input Format**:
```python
# Input: preprocessed DataFrame with normalized features
df_input = pd.DataFrame({
    'Amount_Normalized': [0.75, 0.40, 0.95],
    'Day_of_Week': [0, 1, 5],
    'Month': [1, 1, 1],
    'Category': ['Groceries', 'Transportation', 'Entertainment']
})
```

**Output Format**:
```python
# Returns tuple with three elements:
# 1. DataFrame with cluster labels
clustered_df = pd.DataFrame({
    'Amount_Normalized': [0.75, 0.40, 0.95],
    'Day_of_Week': [0, 1, 5],
    'Month': [1, 1, 1],
    'Category': ['Groceries', 'Transportation', 'Entertainment'],
    'Cluster': [0, 1, 2]  # Assigned cluster labels
})

# 2. Trained KMeans model object
kmeans_model = KMeans(n_clusters=3, random_state=42)

# 3. Cluster summary dictionary
cluster_summary = {
    'cluster_0': {
        'size': 450,
        'avg_amount': 1200.00,
        'dominant_category': 'Groceries',
        'common_days': ['Monday', 'Tuesday', 'Wednesday']
    },
    'cluster_1': {
        'size': 320,
        'avg_amount': 800.00,
        'dominant_category': 'Transportation',
        'common_days': ['Monday', 'Friday']
    }
}
```

**Edge Cases & Assumptions**:
- Assumes minimum 2 clusters for meaningful analysis
- Handles cases where optimal clusters > available data points
- Requires at least 10 transactions for stable clustering
- Assumes features are properly normalized
- Falls back to k=3 if elbow method fails

## D. Troubleshooting Tips

### Common Installation Issues

**Problem**: `ModuleNotFoundError: No module named 'sklearn'`
**Solution**: 
```bash
pip install scikit-learn
# or
conda install scikit-learn
```

**Problem**: Jupyter kernel not found
**Solution**:
```bash
python -m ipykernel install --user --name budget-management
```

### Data Loading Issues

**Problem**: `pandas.errors.ParserError: Error tokenizing data`
**Solution**: 
- Check CSV file encoding (try UTF-8 or latin-1)
- Ensure consistent delimiter usage
- Remove special characters from column names
```python
df = pd.read_csv('data.csv', encoding='utf-8', delimiter=',')
```

**Problem**: Date parsing errors
**Solution**:
```python
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
df = df.dropna(subset=['Date'])  # Remove rows with invalid dates
```

### Clustering Issues

**Problem**: "Empty cluster" warning
**Solution**: 
- Reduce number of clusters
- Check for outliers in the data
- Ensure sufficient data points (>100 transactions recommended)

**Problem**: Poor clustering results
**Solution**:
- Try different feature combinations
- Apply additional data preprocessing
- Consider using different clustering algorithms (DBSCAN, Hierarchical)

### GenAI Integration Issues

**Problem**: OpenAI API key errors
**Solution**:
- Verify API key is correctly set in `.env` file
- Check API key has sufficient credits
- Ensure proper environment variable loading:
```python
from dotenv import load_dotenv
load_dotenv()
```

**Problem**: Rate limiting errors
**Solution**:
- Add delays between API calls
- Implement exponential backoff
- Consider using smaller batch sizes

### Performance Tips

**For Large Datasets (>10,000 transactions)**:
- Use data sampling for initial exploration
- Consider dimensionality reduction (PCA) before clustering
- Implement batch processing for GenAI calls

**Memory Issues**:
- Process data in chunks
- Use `pd.read_csv(chunksize=1000)` for large files
- Clear unnecessary variables with `del variable_name`

### Getting Help

If you encounter issues not covered here:
1. Check the GitHub Issues page for similar problems
2. Ensure all dependencies are correctly installed
3. Verify your data format matches the expected structure
4. Check Python and package versions compatibility
