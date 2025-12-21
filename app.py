import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# KPIs and Page Conig
st.set_page_config(page_title="Customer Invoice Dashboard", layout="wide")

# state sync mechanism
if 'pending_customer_update' in st.session_state:
    st.session_state.customer_selector = [st.session_state.pending_customer_update]
    del st.session_state.pending_customer_update

# File Uploader
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])

@st.cache_data
def load_data(file):
    # Read the file
    # Based on inspection, the first row seems to be the header
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    
    # Filter out rows where 'Type' is empty (this removes section headers and total rows)
    df = df[df['Type'].notna() & (df['Type'] != '')]
    
    # Clean 'Open Balance' column: remove commas and convert to float
    # Based on the file, the 'Open Balance' column might need stripped too
    # Check if 'Open Balance' exists, otherwise look for last column
    if 'Open Balance' not in df.columns:
        # Fallback if column names are messy
        df['Open Balance'] = df.iloc[:, -1]
    
    # Clean numeric columns
    df['Open Balance'] = df['Open Balance'].astype(str).str.replace(',', '').astype(float)
    
    # Parse Date
    # Date format seems to be DD/MM/YYYY based on "26/08/2025"
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df['Due Date'] = pd.to_datetime(df['Due Date'], format='%d/%m/%Y', errors='coerce')
    
    return df

try:
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        st.info("Please upload a CSV file to begin analysis.")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Title and sidebar
# Title was already set above, but we can keep it flowing

# Sidebar Filters
# 1. Customer Filter
customers = sorted(df['Name'].unique().tolist())
selected_customers = st.sidebar.multiselect("Select Customers", customers, key="customer_selector")

# 2. Status/Type Filter
types = sorted(df['Type'].unique().tolist())
selected_types = st.sidebar.multiselect("Select Transaction Type", types, default=[t for t in types if t in ['Invoice', 'Payment']])

# Apply filters
filtered_df = df.copy()
if selected_customers:
    filtered_df = filtered_df[filtered_df['Name'].isin(selected_customers)]
if selected_types:
    filtered_df = filtered_df[filtered_df['Type'].isin(selected_types)]

# KPI Section
total_open_balance = filtered_df['Open Balance'].sum()
total_invoices = len(filtered_df[filtered_df['Type'] == 'Invoice'])
# Calculate count of customers involved
total_customers = filtered_df['Name'].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Total Outstanding Balance", f"{total_open_balance:,.2f}")
col2.metric("Open Invoices Count", f"{total_invoices}")
col3.metric("Active Customers", f"{total_customers}")

# Visualizations
st.markdown("---")
# Visualizations
st.markdown("---")

if selected_customers:
    # --- DETAILED VIEW (Customer Selected) ---
    st.subheader(f"Analysis for Selected Customers")
    
    # Row 1: Chart and Summary Table
    col_chart, col_summary = st.columns([2, 1])
    
    # Calculate Monthly Data
    trend_df = filtered_df.sort_values(by='Date')
    # Use sortable period for sorting, formatted string for display
    trend_df['Month_Sort'] = trend_df['Date'].dt.to_period('M')
    trend_df['Month_Label'] = trend_df['Date'].dt.strftime('%b %Y')
    
    monthly_net = trend_df.groupby(['Month_Sort', 'Month_Label'])['Open Balance'].sum().reset_index()
    monthly_net = monthly_net.sort_values('Month_Sort')
    
    with col_chart:
        st.markdown("#### Monthly Net Transaction Amount")
        # Bar Chart for Monthly Amounts
        fig_month = px.bar(monthly_net, x='Month_Label', y='Open Balance',
                           title="Monthly Net Amount",
                           text_auto='.2s', color='Open Balance')
        fig_month.update_layout(xaxis_title="Month", yaxis_title="Net Amount")
        st.plotly_chart(fig_month, use_container_width=True)
        
    with col_summary:
        st.markdown("#### Monthly Summary")
        # Simple Summary Table
        summary_display = monthly_net[['Month_Label', 'Open Balance']].copy()
        summary_display.columns = ['Month', 'Net Amount']
        
        st.dataframe(
            summary_display.style.background_gradient(cmap="Reds", subset=['Net Amount']).format("{:,.2f}", subset=['Net Amount']),
            width="stretch",
            hide_index=True
        )
        
    # Row 2: Detailed Table
    st.markdown("---")
    st.markdown("#### Detailed Transaction History")
    st.dataframe(
        filtered_df[['Date', 'Type', 'Num', 'Due Date', 'Open Balance']].sort_values(by='Date', ascending=False).style.format({"Open Balance": "{:,.2f}"}),
        width="stretch",
        column_config={
            "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
            "Due Date": st.column_config.DateColumn("Due Date", format="DD/MM/YYYY"),
            "Open Balance": st.column_config.NumberColumn("Amount")
        },
        hide_index=True,
        height=500
    )

else:
    # --- OVERVIEW MODE (No Customer Selected) ---
    st.subheader("Global Visual Analysis")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Balance by Customer")
        # Group by Name and sum Open Balance
        # Let's show Net Open Balance per Customer
        customer_balance = filtered_df.groupby('Name')['Open Balance'].sum().reset_index()
        customer_balance = customer_balance.sort_values(by='Open Balance', ascending=False).head(10) # Top 10
        
        fig_bar = px.bar(customer_balance, x='Name', y='Open Balance', 
                         title="Top 10 Customers by Open Balance",
                         text_auto='.2s', color='Open Balance')
        fig_bar.update_layout(xaxis_title="Customer", yaxis_title="Open Balance")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.markdown("#### Monthly Open Balance Trend")
        # Group by Name and Month for Time Series
        trend_df = filtered_df.sort_values(by='Date')
        trend_df['Month'] = trend_df['Date'].dt.to_period('M').astype(str)
        
        # Sum 'Open Balance' per month (Net change)
        monthly_change = trend_df.groupby(['Name', 'Month'])['Open Balance'].sum().reset_index()
        
        # Calculate Cumulative Sum per Customer to get the Balance at end of month
        monthly_change['Cumulative Balance'] = monthly_change.groupby('Name')['Open Balance'].cumsum()
        
        fig_line = px.line(monthly_change, x='Month', y='Cumulative Balance', color='Name', markers=True,
                           title="Cumulative Balance Trend per Customer")
        st.plotly_chart(fig_line, use_container_width=True)

    # Pivot Table: Customer vs Month
    st.markdown("---")
    st.subheader("Customer Monthly Net Balance Matrix")

    # Prepare data for pivot
    pivot_data = filtered_df.copy()
    # Create a sortable month key and a display label
    pivot_data['Month_Sort'] = pivot_data['Date'].dt.to_period('M')
    pivot_data['Month_Label'] = pivot_data['Date'].dt.strftime('%b %Y')

    # Get the list of months sorted chronologically
    if not pivot_data.empty:
        sorted_months = pivot_data.sort_values('Month_Sort')['Month_Label'].unique()
    
        # Create pivot table
        # Summing Open Balance automatically handles the negative payments
        pivot_table = pivot_data.pivot_table(index='Name', columns='Month_Label', values='Open Balance', aggfunc='sum', fill_value=0)

        # Reindex columns to ensure chronological order
        # Filter sorted_months to only those present in the pivot table (to be safe, though unique() above should match)
        existing_months = [m for m in sorted_months if m in pivot_table.columns]
        pivot_table = pivot_table[existing_months]

        # Add a Total Column
        pivot_table['Total'] = pivot_table.sum(axis=1)

        # Sort by Total descending
        pivot_table = pivot_table.sort_values(by='Total', ascending=False)

        # Reset index to make Name a scrollable column (removes "frozen" behavior that causes covering)
        pivot_table_display = pivot_table.reset_index()

        # Define column configuration
        column_config = {
            "Name": st.column_config.TextColumn("Customer Name", help="Name of the customer"),
            "Total": st.column_config.NumberColumn("Total Balance"),
        }
        # Add config for monthly columns
        for month in existing_months:
            column_config[month] = st.column_config.NumberColumn(month)

        # Format the table for better readability with heatmap coloring
        selection = st.dataframe(
            pivot_table_display.style.background_gradient(cmap="Reds", subset=existing_months + ['Total']).format("{:,.2f}", subset=existing_months + ['Total']),
            width="stretch",
            hide_index=True, # Hide the default numeric index
            column_config=column_config,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        # Handle Selection
        if selection.selection.rows:
            selected_index = selection.selection.rows[0]
            # Since we reset index, we just grab the Name from the updated dataframe row
            selected_customer_name = pivot_table_display.iloc[selected_index]['Name']
            
            # Update the sidebar selection state via a pending key to avoid "modify after instantiation" error
            # Only trigger update if it's different to avoid loops (though on_select usually handles distinct events)
            if 'customer_selector' not in st.session_state or st.session_state.customer_selector != [selected_customer_name]:
                 st.session_state.pending_customer_update = selected_customer_name
                 st.rerun()
    else:
        st.info("No data available for the current filters.")
