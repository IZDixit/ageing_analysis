import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define custom lighter red colormap
# Replacing standard "Reds" which was too dull/dark
lighter_red_cmap = mcolors.LinearSegmentedColormap.from_list("LighterRed", ["#ffffff", "#ff4b4b"])

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
            summary_display.style.background_gradient(cmap=lighter_red_cmap, subset=['Net Amount']).format("{:,.2f}", subset=['Net Amount']),
            use_container_width=True,
            hide_index=True
        )
        
    # Row 2: Detailed Table
    st.markdown("---")
    st.markdown("#### Detailed Transaction History")
    st.dataframe(
        filtered_df[['Date', 'Type', 'Num', 'Due Date', 'Open Balance']].sort_values(by='Date', ascending=False).style.format({"Open Balance": "{:,.2f}"}),
        use_container_width=True,
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

    # --- NEW: Monthly Open Invoices ---
    st.markdown("### Monthly Open Invoices Overview")
    
    # 1. Filter for Open Invoices
    # We care about Invoices that are still open
    monthly_invoices_df = filtered_df[(filtered_df['Type'] == 'Invoice') & (filtered_df['Open Balance'] > 0)].copy()
    
    if not monthly_invoices_df.empty:
        # 2. Group by Month
        monthly_invoices_df['Month_Sort'] = monthly_invoices_df['Date'].dt.to_period('M')
        monthly_invoices_df['Month_Label'] = monthly_invoices_df['Date'].dt.strftime('%b %Y')
        
        # Sum Open Balance per Month
        monthly_grouped = monthly_invoices_df.groupby(['Month_Sort', 'Month_Label'])['Open Balance'].sum().reset_index()
        monthly_grouped = monthly_grouped.sort_values('Month_Sort')
        
        # 3. Visualization - Bar Chart
        fig_monthly_open = px.bar(monthly_grouped, x='Month_Label', y='Open Balance',
                                 title="Total Open Invoices Amount by Month",
                                 text_auto='.2s', color='Open Balance')
        fig_monthly_open.update_layout(xaxis_title="Month", yaxis_title="Total Open Amount")
        fig_monthly_open.update_layout(xaxis_title="Month", yaxis_title="Total Open Amount")
        st.plotly_chart(fig_monthly_open, use_container_width=True)
        
    else:
        st.info("No open invoices found.")
    
    st.markdown("---")
    
    # --- NEW: Interactive Collections Table (Drill-Down) ---
    st.subheader("Interactive Collections Analysis")
    
    # 1. User Input for Period
    group_days = st.number_input("Aggregation Period (Days)", min_value=1, value=30, step=1)
    
    # 2. Prepare Data
    # Use the same 'monthly_invoices_df' (Open Invoices) if available, or filter again
    table_df = filtered_df[(filtered_df['Type'] == 'Invoice') & (filtered_df['Open Balance'] > 0)].copy()
    
    if not table_df.empty:
        # Sort by Date
        table_df = table_df.sort_values('Date')
        
        # Resample logic
        # We need a custom grouper or just resample on Date
        # Calculate a 'Period Group' to group dates into bins of 'group_days'
        # To make it clean, we can assume the periods start from the first date in the dataset
        min_date = table_df['Date'].min()
        
        # Assign a 'Period ID' to each row
        # (Row Date - Min Date) // Days -> Period Index
        table_df['Days_From_Start'] = (table_df['Date'] - min_date).dt.days
        table_df['Period_Index'] = table_df['Days_From_Start'] // group_days
        
        # Calculate Start and End for each period index
        # Start = Min + Index * Days
        # End = Start + Days - 1 (inclusive usually, or just < Start + Days)
        
        # Group by Period Index to get aggregations
        period_stats = table_df.groupby('Period_Index').agg({
            'Open Balance': ['sum', 'count'],
            'Date': 'min' # Just to get a representative date, though we calculate exact start below
        }).reset_index()
        
        # Flatten columns
        period_stats.columns = ['Period_Index', 'Total Amount', 'Invoice Count', 'First_Date_In_Group']
        
        # Calculate Display Columns
        period_stats['Period Start'] = min_date + pd.to_timedelta(period_stats['Period_Index'] * group_days, unit='D')
        period_stats['Period End'] = period_stats['Period Start'] + pd.to_timedelta(group_days - 1, unit='D')
        
        # Create Label
        period_stats['Period Label'] = period_stats['Period Start'].dt.strftime('%Y-%m-%d') + " to " + period_stats['Period End'].dt.strftime('%Y-%m-%d')
        
        # Format for display
        display_table = period_stats[['Period Start', 'Period End', 'Invoice Count', 'Total Amount']].copy()
        
        # 3. Display Summary Table
        st.markdown("##### Summary by Period")
        st.write("Select a row to see details.")
        
        selection = st.dataframe(
            display_table.style.format({"Total Amount": "{:,.2f}"}),
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun"
        )
        
        # 4. Handle Selection (Drill Down)
        if selection.selection.rows:
            selected_idx = selection.selection.rows[0]
            # Get the Period Start/End from the displayed dataframe
            # Note: The displayed dataframe index matches logic if we didn't sort differently. 
            # We strictly constructed `display_table` from `period_stats` which is sorted by Index.
            # So `selected_idx` maps directly to `period_stats` row.
            
            sel_record = period_stats.iloc[selected_idx]
            p_start = sel_record['Period Start']
            p_end = sel_record['Period End']
            
            st.markdown(f"##### Details for Period: {p_start.strftime('%d/%m/%Y')} - {p_end.strftime('%d/%m/%Y')}")
            
            # Filter Original Data
            # Start is inclusive, End is inclusive (since we successfully subtracted 1 day for display, let's use the explicit comparison)
            # Actually, using strictly date comparison:
            drill_down_df = table_df[
                (table_df['Date'] >= p_start) & 
                (table_df['Date'] <= p_start + pd.Timedelta(days=group_days)) # Use upper bound exclusive-like logic or cover the full window
            ].copy()
            
            # Wait, `Period End` above was `Start + Days - 1`. 
            # So `[Start, End]` is the inclusive range.
            drill_down_df = table_df[
                (table_df['Date'] >= p_start) & 
                (table_df['Date'] <= p_end + pd.Timedelta(days=1)) # slightly loose to capture time components if any, though we loaded as date. Safe to use <= End if pure dates.
            ]
            
            # Show Detailed Table
            st.dataframe(
                drill_down_df[['Name', 'Date', 'Due Date', 'Open Balance', 'Num']],
                use_container_width=True,
                column_config={
                    "Date": st.column_config.DateColumn("Inv Date", format="DD/MM/YYYY"),
                    "Due Date": st.column_config.DateColumn("Due Date", format="DD/MM/YYYY"),
                    "Open Balance": st.column_config.NumberColumn("Amount", format="$%.2f"),
                    "Name": "Customer"
                },
                hide_index=True
            )
    else:
        st.info("No data for table.")
    
    st.markdown("---")

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
            pivot_table_display.style.background_gradient(cmap=lighter_red_cmap, subset=existing_months + ['Total']).format("{:,.2f}", subset=existing_months + ['Total']),
            use_container_width=True,
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
