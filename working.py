import streamlit as st
import pandas as pd
import numpy as np
import simpy
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime, timedelta

# ------------------------------------------------
# Page and Session Setup
# ------------------------------------------------
st.set_page_config(page_title="Supermarket POS Customer Simulation", layout="wide")
st.title("Supermarket POS Customer Simulation (Time-based or Customer-based)")

# Initialize Session State for Simulation Results
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'customers' not in st.session_state:
    st.session_state.customers = None
if 'wait_times' not in st.session_state:
    st.session_state.wait_times = None
if 'service_times_sim' not in st.session_state:
    st.session_state.service_times_sim = None
if 'total_times' not in st.session_state:
    st.session_state.total_times = None

# ------------------------------------------------
# Data Loading and Processing
# ------------------------------------------------
@st.cache_data
def load_data():
    """
    Reads pos_data.csv and relies on the file's own 'waiting_time' and 'service_time' columns,
    which already have the correct average wait and service times.
    """
    try:
        df = pd.read_csv("./working_data.csv")
    except FileNotFoundError:
        st.error("The file './working_data.csv' was not found.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while reading the data: {e}")
        st.stop()

    # 1. Rename the file's columns for clarity (optional).
    #    For example, if your CSV has these columns exactly:
    #       'waiting_time', 'service_time', 'arrival in integer', 'service_start', 'service_end', etc.
    #    we can rename them so we don't confuse them with the new columns we create.
    rename_map = {
        "waiting_time": "wait_time_csv",    # CSV's existing wait times
        "service_time": "service_time_csv"  # CSV's existing service times
    }
    for col in rename_map:
        if col not in df.columns:
            st.error(f"Column '{col}' not found in the CSV. Please verify your column names.")
            st.stop()

    df.rename(columns=rename_map, inplace=True)

    # 2. Use the CSV's own wait_time and service_time as the main columns.
    #    This ensures you get the 7.7 average (or whatever your CSV truly has).
    df["wait_time"] = df["wait_time_csv"]
    df["service_time"] = df["service_time_csv"]

    # (Optional) If you want to see the numeric columns for arrival, start, end:
    # df.rename(columns={
    #     "arrival in integer": "arrival_min",
    #     "service_start": "start_min",
    #     "service_end": "end_min",
    # }, inplace=True)
    # ... but we won't recompute wait_time or service_time from them.

    # 3. Fill any NaN with 0 just to be safe.
    df["wait_time"] = df["wait_time"].fillna(0)
    df["service_time"] = df["service_time"].fillna(0)

    return df

data = load_data()

# ------------------------------------------------
# Sidebar: Simulation Parameters
# ------------------------------------------------
st.sidebar.header("Simulation Parameters")

# 1. Queuing Model Selection
model_options = ["M/M/1", "M/M/c", "M/G/1"]
model_choice = st.sidebar.selectbox("Select Queuing Model", model_options, key="queuing_model")

# 2. Number of Servers (Force 1 for M/M/1 and M/G/1)
if model_choice in ["M/M/1", "M/G/1"]:
    st.sidebar.info("For this model, number of servers is fixed at 1.")
    num_servers = 1
else:
    num_servers = st.sidebar.number_input(
        "Number of Servers", 
        min_value=1, 
        max_value=10, 
        value=2, 
        step=1,
        key="num_servers"
    )

# 3. Random Seed (optional)
seed = st.sidebar.number_input(
    "Random Seed (0 for random)", 
    min_value=0, 
    max_value=10000, 
    value=0, 
    step=1,
    key="random_seed"
)
if seed != 0:
    np.random.seed(seed)

# 4. Arrival Distribution
arrival_dist_options = ["Exponential", "Uniform", "Normal"]
arrival_dist_choice = st.sidebar.selectbox("Arrival Time Distribution", arrival_dist_options, key="arrival_distribution")
st.sidebar.subheader("Inter Arrival Distribution Parameters")
if arrival_dist_choice == "Exponential":
    arrival_lambda = st.sidebar.number_input(
        "Lambda (λ)", 
        min_value=0.1, 
        max_value=10.0, 
        value=0.5, 
        step=0.1,
        key="arrival_lambda_exponential"
    )
elif arrival_dist_choice == "Uniform":
    arrival_low = st.sidebar.number_input(
        "Low (min)", 
        min_value=0.0, 
        max_value=10.0, 
        value=1.0, 
        step=0.1,
        key="arrival_low_uniform"
    )
    arrival_high = st.sidebar.number_input(
        "High (max)", 
        min_value=0.0, 
        max_value=20.0, 
        value=5.0, 
        step=0.1,
        key="arrival_high_uniform"
    )
    if arrival_high <= arrival_low:
        st.sidebar.error("High must be greater than Low for Uniform Distribution.")
elif arrival_dist_choice == "Normal":
    arrival_mu = st.sidebar.number_input(
        "Mean (μ)", 
        min_value=0.0, 
        max_value=20.0, 
        value=5.0, 
        step=0.1,
        key="arrival_mu_normal"
    )
    arrival_sigma = st.sidebar.number_input(
        "Standard Deviation (σ)", 
        min_value=0.1, 
        max_value=10.0, 
        value=1.0, 
        step=0.1,
        key="arrival_sigma_normal"
    )

# 5. Service Distribution
service_dist_options = ["Exponential", "Uniform", "Normal"]
service_dist_choice = st.sidebar.selectbox("Service Time Distribution", service_dist_options, key="service_distribution")
st.sidebar.subheader("Service Distribution Parameters")
if service_dist_choice == "Exponential":
    service_lambda = st.sidebar.number_input(
        "Lambda (λ)", 
        min_value=0.1, 
        max_value=10.0, 
        value=0.5, 
        step=0.1,
        key="service_lambda_exponential"
    )
elif service_dist_choice == "Uniform":
    service_low = st.sidebar.number_input(
        "Low (min)", 
        min_value=0.0, 
        max_value=20.0, 
        value=2.0, 
        step=0.1,
        key="service_low_uniform"
    )
    service_high = st.sidebar.number_input(
        "High (max)", 
        min_value=0.0, 
        max_value=30.0, 
        value=5.0, 
        step=0.1,
        key="service_high_uniform"
    )
    if service_high <= service_low:
        st.sidebar.error("High must be greater than Low for Uniform Distribution.")
elif service_dist_choice == "Normal":
    service_mu = st.sidebar.number_input(
        "Mean (μ)", 
        min_value=0.0, 
        max_value=30.0, 
        value=5.0, 
        step=0.1,
        key="service_mu_normal"
    )
    service_sigma = st.sidebar.number_input(
        "Standard Deviation (σ)", 
        min_value=0.1, 
        max_value=15.0, 
        value=1.0, 
        step=0.1,
        key="service_sigma_normal"
    )

# 6. Simulation Approach
run_mode = st.sidebar.radio(
    "Generate Arrivals By:",
    ("Time-based", "Customer-based"),
    index=0,
    key="run_mode"
)

# 7. If Time-based, get a maximum simulation time
if run_mode == "Time-based":
    sim_time = st.sidebar.number_input(
        "Total Simulation Arrival Period (minutes)", 
        min_value=10, 
        max_value=1000, 
        value=60, 
        step=10,
        key="simulation_time"
    )
else:
    sim_time = 0  # Not used in customer-based mode

# 8. If Customer-based, get the number of customers to generate
if run_mode == "Customer-based":
    num_customers = st.sidebar.number_input(
        "Number of Customers",
        min_value=1,
        max_value=10000,
        value=50,
        step=1,
        key="num_customers"
    )
else:
    num_customers = 0  # Not used in time-based mode

# 9. Run Simulation Button
run_sim = st.sidebar.button("Run Simulation", key="run_simulation_button")

# ------------------------------------------------
# Distribution Functions
# ------------------------------------------------
def get_interarrival_time():
    """Return a single inter-arrival time from the chosen distribution."""
    if arrival_dist_choice == "Exponential":
        return np.random.exponential(1 / arrival_lambda)
    elif arrival_dist_choice == "Uniform":
        return np.random.uniform(arrival_low, arrival_high)
    elif arrival_dist_choice == "Normal":
        return max(0, np.random.normal(arrival_mu, arrival_sigma))

def get_service_time():
    """Return a single service time from the chosen distribution."""
    if service_dist_choice == "Exponential":
        return np.random.exponential(1 / service_lambda)
    elif service_dist_choice == "Uniform":
        return np.random.uniform(service_low, service_high)
    elif service_dist_choice == "Normal":
        return max(0, np.random.normal(service_mu, service_sigma))

# ------------------------------------------------
# Simulation Environment
# ------------------------------------------------
class Customer:
    """Represents a customer in the simulation."""
    def __init__(self, env, name, server_store):
        self.env = env
        self.name = name
        self.server_store = server_store
        self.server_id = None
        self.wait_time = 0
        self.service_time = 0
        self.start_time = 0
        self.end_time = 0

    def process(self):
        arrival_time = self.env.now
        # Request a server (get a server ID)
        server_id = yield self.server_store.get()
        self.server_id = server_id
        self.start_time = self.env.now
        self.wait_time = self.start_time - arrival_time
        # Determine service time from chosen distribution
        service_time = get_service_time()
        self.service_time = service_time
        yield self.env.timeout(service_time)
        self.end_time = self.env.now
        # Release server back to the store
        yield self.server_store.put(self.server_id)

def init_server_store(env, server_store, n_servers):
    """Initialize the server store with server IDs (1..N)."""
    for i in range(1, n_servers + 1):
        yield server_store.put(i)

def create_customer(env, i, arrival_time, server_store, customers):
    """Create a single customer after waiting until arrival_time."""
    yield env.timeout(arrival_time - env.now)
    customer = Customer(env, f"C{i}", server_store)
    customers.append(customer)
    env.process(customer.process())

def run_simulation():
    """Run the simulation with the chosen approach, distribution, and parameters."""
    env = simpy.Environment()
    server_store = simpy.Store(env, capacity=num_servers)
    env.process(init_server_store(env, server_store, num_servers))

    customers = []

    # ---------------------------
    # TIME-BASED approach
    # ---------------------------
    if run_mode == "Time-based":
        arrival_times = [0]  # Start with a customer at time=0
        while arrival_times[-1] < sim_time:
            inter_arrival = get_interarrival_time()
            next_arrival = arrival_times[-1] + inter_arrival
            if next_arrival > sim_time:
                break
            arrival_times.append(next_arrival)

    # ---------------------------
    # CUSTOMER-BASED approach
    # ---------------------------
    else:  # run_mode == "Customer-based"
        arrival_times = [0]
        for _ in range(num_customers - 1):
            inter_arrival = get_interarrival_time()
            next_arrival = arrival_times[-1] + inter_arrival
            arrival_times.append(next_arrival)

    # Create customer processes
    for i, arrival in enumerate(arrival_times, start=1):
        env.process(create_customer(env, i, arrival, server_store, customers))

    # Run the simulation until completion
    env.run()
    total_simulation_time = env.now

    # Collect metrics
    wait_times = [c.wait_time for c in customers]
    service_times_sim = [c.service_time for c in customers]
    total_times = [c.wait_time + c.service_time for c in customers]

    total_busy_time = sum(service_times_sim)
    utilization = (total_busy_time) / (num_servers * total_simulation_time) * 100 if total_simulation_time > 0 else 0

    # Number of customers served
    num_cust = len(customers)

    # Compute additional metrics
    L = num_cust / total_simulation_time if total_simulation_time > 0 else 0  # Avg number in system
    Lq = sum(wait_times) / total_simulation_time if total_simulation_time > 0 else 0  # Avg queue length
    W = np.mean(total_times) if total_times else 0  # Avg total time in system
    Wq = np.mean(wait_times) if wait_times else 0   # Avg wait time

    # Arrival rate
    # - For time-based, this is num_cust / sim_time
    # - For customer-based, define it as num_cust / total_simulation_time
    if run_mode == "Time-based":
        arrival_rate = num_cust / sim_time if sim_time else 0
    else:
        arrival_rate = num_cust / total_simulation_time if total_simulation_time else 0

    # Service rate
    service_rate = num_cust / sum(service_times_sim) if sum(service_times_sim) > 0 else 0

    metrics = {
        "System Efficiency (%)": round(utilization, 2),
        "System Idle Time (%)": round(100 - utilization, 2),
        "System L (Avg Number in System)": round(L, 2),
        "System Lq (Avg Queue Length)": round(Lq, 2),
        "System W (Avg Total Time in System)": round(W, 2),
        "System Wq (Avg Wait Time)": round(Wq, 2),
        "System λ (Arrival Rate)": round(arrival_rate, 2),
        "System μ (Service Rate)": round(service_rate, 2),
        "System ρ (Overall Utilization)": round(utilization/100, 2),
        "Total Customers Served": num_cust,
        "Total Servers": num_servers,
        "Total Simulation Time (minutes)": round(total_simulation_time, 2)
    }

    return metrics, wait_times, service_times_sim, total_times, customers

# ------------------------------------------------
# Run Simulation and Display Results
# ------------------------------------------------
if run_sim:
    with st.spinner("Running simulation..."):
        metrics, wait_times, service_times_sim, total_times, customers = run_simulation()
        st.session_state.metrics = metrics
        st.session_state.wait_times = wait_times
        st.session_state.service_times_sim = service_times_sim
        st.session_state.total_times = total_times
        st.session_state.customers = customers
    st.success("Simulation Completed!")

if st.session_state.metrics:
    # Display Simulation Metrics
    st.subheader("Simulation Metrics")
    metrics_df = pd.DataFrame(st.session_state.metrics.items(), columns=["Metric", "Value"])
    st.table(metrics_df)

    # Service Time Distribution Plot
    st.subheader("Service Time Distribution (Simulation)")
    if st.session_state.service_times_sim:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(st.session_state.service_times_sim, bins=20, kde=True, ax=ax, color='skyblue')
        ax.set_xlabel("Service Time (minutes)")
        ax.set_ylabel("Frequency")
        ax.set_title("Service Time Distribution (Simulation)")
        st.pyplot(fig)
    else:
        st.write("No service time data available for visualization.")

    # Optional Gantt Chart
    if st.checkbox("Show Gantt Chart"):
        st.subheader("Gantt Chart of Customer Service (Simulation)")
        try:
            gantt_data = []
            base_time = datetime(2024, 1, 1, 8, 0, 0)  # Arbitrary reference
            for c in st.session_state.customers:
                if c.end_time >= c.start_time:
                    start_time = base_time + timedelta(minutes=c.start_time)
                    finish_time = base_time + timedelta(minutes=c.end_time)
                    gantt_data.append({
                        "Task": c.name,
                        "Start": start_time,
                        "Finish": finish_time,
                        "Resource": f"Server {c.server_id}"
                    })

            if gantt_data:
                df_gantt = pd.DataFrame(gantt_data)
                fig_gantt = px.timeline(
                    df_gantt, 
                    x_start='Start', 
                    x_end='Finish', 
                    y='Resource', 
                    color='Resource', 
                    hover_name='Task',
                    title='Gantt Chart of Customer Service'
                )
                fig_gantt.update_yaxes(categoryorder='total ascending')
                fig_gantt.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Server",
                    title="Gantt Chart of Customer Service",
                    showlegend=False
                )
                st.plotly_chart(fig_gantt, use_container_width=True)
            else:
                st.write("No data available for Gantt Chart.")
        except Exception as e:
            st.error(f"An error occurred while generating the Gantt Chart: {e}")

    # Compare Simulation Means vs. Real Data Means
    st.subheader("Data Mean Comparison (Real vs. Simulation)")
    try:
        # Because we replaced df["wait_time"] with the CSV's existing wait_time,
        # data["wait_time"].mean() should now match your Excel average (~7.7).
        data_mean_wait = data["wait_time"].mean()
        data_mean_service = data["service_time"].mean()

        if np.isnan(data_mean_wait) or np.isnan(data_mean_service):
            st.error("Cannot compute mean wait_time or service_time from the data.")
        else:
            sim_mean_wait = np.mean(st.session_state.wait_times) if st.session_state.wait_times else 0
            sim_mean_service = np.mean(st.session_state.service_times_sim) if st.session_state.service_times_sim else 0

            comparison_df = pd.DataFrame({
                "Metric": ["Average Wait Time (minutes)", "Average Service Time (minutes)"],
                "Simulation": [round(sim_mean_wait, 2), round(sim_mean_service, 2)],
                "Real Data": [round(data_mean_wait, 2), round(data_mean_service, 2)]
            })
            st.table(comparison_df)

            fig_comparison = px.bar(
                comparison_df.melt(id_vars="Metric", var_name="Source", value_name="Value"),
                x="Metric",
                y="Value",
                color="Source",
                barmode='group',
                title="Comparison of Average Times"
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred during Data Mean Comparison: {e}")

else:
    st.write("Adjust the simulation parameters in the sidebar and click 'Run Simulation'.")

# ------------------------------------------------
# Optionally Display Original Data
# ------------------------------------------------
if st.checkbox("Show Original POS Data"):
    st.subheader("POS Customer Data (Using CSV's own wait & service times)")
    st.dataframe(data)
