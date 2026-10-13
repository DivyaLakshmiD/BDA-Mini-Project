from pathlib import Path
import ast
import re

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

LIVE_PREDICTIONS_DIR = BASE_DIR / "outputs" / "live_predictions"

st.set_page_config(
    page_title="Predictive Maintenance | Factory Intelligence",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 16px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


def read_csv_file(path, required=True):
    if not path.exists():
        if required:
            st.error(f"Required file not found: {path}")
            st.stop()
        return pd.DataFrame()
    return pd.read_csv(path)


def parse_failure_probability(value):
    if pd.isna(value):
        return 0.0

    if isinstance(value, (list, tuple)):
        return float(value[1]) if len(value) > 1 else 0.0

    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, (list, tuple)) and len(parsed) > 1:
            return float(parsed[1])
    except (ValueError, SyntaxError, TypeError):
        pass

    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", str(value))
    if len(numbers) >= 2:
        return float(numbers[1])

    return 0.0


@st.cache_data(show_spinner="Loading factory datasets...")
def load_data():
    predictions = read_csv_file(RESULTS_DIR / "test_predictions.csv")
    telemetry = read_csv_file(DATA_DIR / "PdM_telemetry.csv", required=False)
    machines = read_csv_file(DATA_DIR / "PdM_machines.csv", required=False)
    maintenance = read_csv_file(DATA_DIR / "PdM_maint.csv", required=False)
    failures = read_csv_file(DATA_DIR / "PdM_failures.csv", required=False)
    errors = read_csv_file(DATA_DIR / "PdM_errors.csv", required=False)

    predictions.columns = predictions.columns.str.strip()
    predictions["datetime"] = pd.to_datetime(
        predictions["datetime"], errors="coerce"
    )
    predictions["machineID"] = pd.to_numeric(
        predictions["machineID"], errors="coerce"
    )
    predictions["prediction"] = pd.to_numeric(
        predictions["prediction"], errors="coerce"
    ).fillna(0).astype(int)
    predictions["failure_within_24h"] = pd.to_numeric(
        predictions["failure_within_24h"], errors="coerce"
    ).fillna(0).astype(int)
    predictions["failure_probability"] = predictions["probability"].apply(
        parse_failure_probability
    )

    for df in [telemetry, maintenance, failures, errors]:
        if not df.empty:
            df.columns = df.columns.str.strip()
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(
                    df["datetime"], errors="coerce"
                )
            if "machineID" in df.columns:
                df["machineID"] = pd.to_numeric(
                    df["machineID"], errors="coerce"
                )

    if not machines.empty:
        machines.columns = machines.columns.str.strip()
        if "machineID" in machines.columns:
            machines["machineID"] = pd.to_numeric(
                machines["machineID"], errors="coerce"
            )

    predictions = predictions.dropna(
        subset=["machineID", "datetime"]
    )
    predictions["machineID"] = predictions["machineID"].astype(int)

    return predictions, telemetry, machines, maintenance, failures, errors


predictions, telemetry, machines, maintenance, failures, errors = load_data()


def load_live_predictions():
    files = sorted(LIVE_PREDICTIONS_DIR.glob("part-*.csv"))

    if not files:
        return pd.DataFrame()

    frames = [pd.read_csv(file) for file in files]
    live = pd.concat(frames, ignore_index=True)

    live.columns = live.columns.str.strip()
    live["datetime"] = pd.to_datetime(
        live["datetime"], errors="coerce", utc=True
    )
    live["machineID"] = pd.to_numeric(
        live["machineID"], errors="coerce"
    )
    live["failure_probability"] = pd.to_numeric(
        live["failure_probability"], errors="coerce"
    )

    live = live.dropna(
        subset=["machineID", "datetime", "failure_probability"]
    )

    if "risk_level" in live.columns:
        live["risk_level"] = live["risk_level"].astype(str).str.upper()

    return live


live_predictions = load_live_predictions()

if predictions.empty:
    st.error("The prediction file contains no usable records.")
    st.stop()

st.title("⚙️ Factory Intelligence")
st.subheader("Predictive Maintenance Monitoring System")

st.caption(
    "Machine failure risk monitoring, equipment history, and maintenance prioritization"
)

st.info(
    "This first version displays historical test-set predictions from your trained "
    "Spark ML model. It is not a live sensor-stream dashboard yet."
)

with st.sidebar:
    st.header("Dashboard Settings")
    st.caption("Adjust risk levels for dashboard monitoring.")

    high_threshold = st.slider(
        "High-risk threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    medium_threshold = st.slider(
        "Medium-risk threshold",
        min_value=0.01,
        max_value=0.80,
        value=0.20,
        step=0.05
    )

    if medium_threshold >= high_threshold:
        st.warning("The medium-risk threshold should be lower than the high-risk threshold.")

    st.divider()
    st.write("**Project pipeline**")
    st.write("Historical data → Spark ML predictions → Dashboard")
    st.caption("Kafka, Hive, and live streaming can be integrated in the next phase.")

latest_predictions = (
    predictions.sort_values("datetime")
    .drop_duplicates(subset=["machineID"], keep="last")
    .copy()
)

if medium_threshold >= high_threshold:
    medium_threshold = max(0.01, high_threshold - 0.05)

def risk_level(probability):
    if probability >= high_threshold:
        return "High"
    if probability >= medium_threshold:
        return "Medium"
    return "Low"


latest_predictions["risk_level"] = latest_predictions[
    "failure_probability"
].apply(risk_level)

if not machines.empty and "machineID" in machines.columns:
    machine_count = machines["machineID"].nunique()
else:
    machine_count = latest_predictions["machineID"].nunique()

high_risk_count = int((latest_predictions["risk_level"] == "High").sum())
medium_risk_count = int((latest_predictions["risk_level"] == "Medium").sum())
predicted_failure_count = int((latest_predictions["prediction"] == 1).sum())

if not predictions.empty:
    test_start = predictions["datetime"].min()
    test_end = predictions["datetime"].max()
else:
    test_start = pd.NaT
    test_end = pd.NaT

st.caption(
    f"Prediction window: {test_start:%d %b %Y} – {test_end:%d %b %Y}"
    if pd.notna(test_start) and pd.notna(test_end)
    else "Prediction window unavailable"
)


tab_overview, tab_monitor, tab_history, tab_schedule, tab_live, tab_model = st.tabs([
    "Factory Overview",
    "Machine Monitoring",
    "Machinery History",
    "Maintenance Priorities",
    "Kafka Predictions",
    "Model Results"
])

with tab_overview:
    st.subheader("Factory Overview")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Machines", f"{machine_count:,}")
    k2.metric("High-Risk Machines", f"{high_risk_count:,}")
    k3.metric("Medium-Risk Machines", f"{medium_risk_count:,}")
    k4.metric("Predicted Failures", f"{predicted_failure_count:,}")

    left, right = st.columns(2)

    with left:
        risk_counts = (
            latest_predictions["risk_level"]
            .value_counts()
            .reindex(["High", "Medium", "Low"], fill_value=0)
            .rename_axis("Risk Level")
            .reset_index(name="Machines")
        )
        fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Machines",
            title="Machine Risk Distribution",
            hole=0.55,
            color="Risk Level",
            color_discrete_map={
                "High": "#E45756",
                "Medium": "#F2A541",
                "Low": "#3A9D78"
            }
        )
        fig.update_layout(legend_title_text="Risk")
        st.plotly_chart(fig, width="stretch")

    with right:
        timeline = (
            predictions.groupby("datetime", as_index=False)
            .agg(
                predicted_failures=("prediction", "sum"),
                actual_failures=("failure_within_24h", "sum")
            )
            .sort_values("datetime")
        )
        fig = px.line(
            timeline,
            x="datetime",
            y=["predicted_failures", "actual_failures"],
            title="Predicted vs Actual Failures Over Time",
            labels={
                "datetime": "Date and time",
                "value": "Number of failures",
                "variable": "Series"
            }
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Machines Requiring Attention")

    attention = latest_predictions[
        latest_predictions["risk_level"].isin(["High", "Medium"])
    ].sort_values("failure_probability", ascending=False)

    if attention.empty:
        st.success(
            "No machines currently meet the selected medium- or high-risk thresholds."
        )
    else:
        display_attention = attention[
            [
                "machineID",
                "datetime",
                "failure_probability",
                "prediction",
                "risk_level"
            ]
        ].copy()
        display_attention["failure_probability"] = (
            display_attention["failure_probability"] * 100
        ).round(2)
        display_attention = display_attention.rename(columns={
            "machineID": "Machine ID",
            "datetime": "Latest Prediction Time",
            "failure_probability": "Failure Risk (%)",
            "prediction": "Predicted Failure",
            "risk_level": "Risk Level"
        })
        st.dataframe(display_attention, width="stretch", hide_index=True)

with tab_monitor:
    st.subheader("Machine Risk Monitoring")

    machine_ids = sorted(latest_predictions["machineID"].dropna().unique().tolist())

    selected_machine = st.selectbox(
        "Select a machine",
        machine_ids,
        format_func=lambda x: f"Machine {int(x)}"
    )

    selected_row = latest_predictions[
        latest_predictions["machineID"] == selected_machine
    ].iloc[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Machine ID", int(selected_machine))
    m2.metric(
        "Failure Probability",
        f"{selected_row['failure_probability'] * 100:.2f}%"
    )
    m3.metric("Risk Level", selected_row["risk_level"])

    if not machines.empty and "machineID" in machines.columns:
        machine_info = machines[
            machines["machineID"] == selected_machine
        ]
        if not machine_info.empty:
            st.markdown("**Machine Information**")
            st.dataframe(machine_info, width="stretch", hide_index=True)

    if not telemetry.empty and "machineID" in telemetry.columns:
        machine_telemetry = telemetry[
            telemetry["machineID"] == selected_machine
        ].sort_values("datetime")

        if not machine_telemetry.empty:
            st.markdown("**Sensor Trends**")

            sensor_columns = [
                col for col in ["volt", "rotate", "pressure", "vibration"]
                if col in machine_telemetry.columns
            ]

            if sensor_columns:
                sensor_choice = st.selectbox(
                    "Choose a sensor",
                    sensor_columns,
                    format_func=lambda x: {
                        "volt": "Voltage",
                        "rotate": "Rotation",
                        "pressure": "Pressure",
                        "vibration": "Vibration"
                    }.get(x, x)
                )

                recent_telemetry = machine_telemetry.tail(500)

                fig = px.line(
                    recent_telemetry,
                    x="datetime",
                    y=sensor_choice,
                    title=f"{sensor_choice.title()} History — Machine {int(selected_machine)}",
                    labels={
                        "datetime": "Date and time",
                        sensor_choice: sensor_choice.title()
                    }
                )
                st.plotly_chart(fig, width="stretch")

                st.caption(
                    "Sensor chart shows up to the latest 500 available records "
                    "for the selected machine."
                )
                st.dataframe(
                    recent_telemetry.tail(20).sort_values(
                        "datetime", ascending=False
                    ),
                    width="stretch",
                    hide_index=True
                )
        else:
            st.warning("No telemetry records were found for this machine.")
    else:
        st.warning("The telemetry dataset is unavailable.")

with tab_history:
    st.subheader("Machinery History")

    history_machine = st.selectbox(
        "Choose a machine for its history",
        machine_ids,
        key="history_machine",
        format_func=lambda x: f"Machine {int(x)}"
    )

    history_sections = [
        ("Failure Records", failures, "failure"),
        ("Maintenance Records", maintenance, "maintenance"),
        ("Error Records", errors, "error")
    ]

    for section_title, source_df, section_type in history_sections:
        st.markdown(f"**{section_title}**")

        if source_df.empty or "machineID" not in source_df.columns:
            st.caption(f"{section_title} dataset is not available.")
            continue

        machine_history = source_df[
            source_df["machineID"] == history_machine
        ].copy()

        if machine_history.empty:
            st.write("No records found for this machine.")
        else:
            if "datetime" in machine_history.columns:
                machine_history = machine_history.sort_values(
                    "datetime", ascending=False
                )
            st.dataframe(
                machine_history.head(100),
                width="stretch",
                hide_index=True
            )
            st.caption(
                f"Showing up to 100 records out of {len(machine_history):,} "
                f"available records."
            )


with tab_schedule:
    st.subheader("Predictive Maintenance Schedule")
    st.write(
        "Proposed maintenance dates are generated from predicted failure risk. "
        "They are recommendations, not confirmed factory bookings."
    )

    schedule = latest_predictions.sort_values(
        "failure_probability", ascending=False
    ).copy()

    if not machines.empty and "machineID" in machines.columns:
        machine_details = machines.drop_duplicates(subset=["machineID"])
        schedule = schedule.merge(
            machine_details,
            on="machineID",
            how="left",
            suffixes=("", "_machine")
        )

    if not maintenance.empty and "machineID" in maintenance.columns:
        if "datetime" in maintenance.columns:
            last_maintenance = (
                maintenance.dropna(subset=["datetime"])
                .sort_values("datetime")
                .drop_duplicates(subset=["machineID"], keep="last")
                [["machineID", "datetime"]]
                .rename(columns={"datetime": "last_maintenance_date"})
            )
            schedule = schedule.merge(
                last_maintenance,
                on="machineID",
                how="left"
            )

    schedule["risk_level"] = (
        schedule["risk_level"].astype(str).str.upper()
    )

    schedule["recommended_action"] = schedule["risk_level"].map({
        "HIGH": "Urgent inspection and maintenance assessment",
        "MEDIUM": "Preventive inspection",
        "LOW": "Routine inspection and continued monitoring"
    }).fillna("Inspect machine and review risk")

    schedule["priority_order"] = schedule["risk_level"].map({
        "HIGH": 1,
        "MEDIUM": 2,
        "LOW": 3
    }).fillna(4)

    schedule = schedule.sort_values(
        ["priority_order", "failure_probability"],
        ascending=[True, False]
    ).reset_index(drop=True)

    schedule["schedule_date"] = ""
    schedule["schedule_slot"] = ""

    slot_hours = ["09:00–10:00", "10:00–11:00", "11:00–12:00",
                  "13:00–14:00", "14:00–15:00", "15:00–16:00"]

    today = pd.Timestamp.today().normalize()
    day_offset = 0
    slot_index = 0

    for i in schedule.index:
        risk = schedule.at[i, "risk_level"]

        if risk == "HIGH":
            earliest_day = 0
        elif risk == "MEDIUM":
            earliest_day = 1
        else:
            earliest_day = 3

        day_offset = max(day_offset, earliest_day)

        if slot_index >= len(slot_hours):
            day_offset += 1
            slot_index = 0

        while (today + pd.Timedelta(days=day_offset)).weekday() >= 5:
            day_offset += 1

        schedule.at[i, "schedule_date"] = (
            today + pd.Timedelta(days=day_offset)
        ).strftime("%Y-%m-%d")

        schedule.at[i, "schedule_slot"] = slot_hours[slot_index]

        slot_index += 1

    schedule["schedule_status"] = "Proposed"

    st.caption(
        "Scheduling assumptions: one machine per time slot, six slots per "
        "working day, weekends skipped. Actual availability is not verified."
    )

    total_machines = schedule["machineID"].nunique()
    high_count = (schedule["risk_level"] == "HIGH").sum()
    medium_count = (schedule["risk_level"] == "MEDIUM").sum()
    low_count = (schedule["risk_level"] == "LOW").sum()

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Machines", total_machines)
    metric2.metric("High Risk", int(high_count))
    metric3.metric("Medium Risk", int(medium_count))
    metric4.metric("Low Risk", int(low_count))

    schedule_display_columns = [
        "machineID",
        "risk_level",
        "failure_probability",
        "schedule_date",
        "schedule_slot",
        "recommended_action",
        "schedule_status"
    ]

    for col in [
        "model",
        "age",
        "last_maintenance_date",
        "datetime"
    ]:
        if col in schedule.columns:
            schedule_display_columns.append(col)

    schedule_view = schedule[schedule_display_columns].copy()
    schedule_view["failure_probability"] = (
        schedule_view["failure_probability"] * 100
    ).round(2)

    schedule_view = schedule_view.rename(columns={
        "machineID": "Machine ID",
        "risk_level": "Risk Level",
        "failure_probability": "Failure Risk (%)",
        "schedule_date": "Recommended Date",
        "schedule_slot": "Proposed Time Slot",
        "recommended_action": "Recommended Action",
        "schedule_status": "Schedule Status",
        "model": "Machine Model",
        "age": "Machine Age",
        "last_maintenance_date": "Last Maintenance",
        "datetime": "Prediction Timestamp"
    })

    st.dataframe(schedule_view, width="stretch", hide_index=True)

    st.download_button(
        "Download Maintenance Schedule CSV",
        data=schedule_view.to_csv(index=False).encode("utf-8"),
        file_name="predictive_maintenance_schedule.csv",
        mime="text/csv"
    )


with tab_live:
    st.subheader("Kafka Sensor Stream Predictions")

    if live_predictions.empty:
        st.warning(
            "No saved Kafka predictions found. Run the prediction pipeline first."
        )
    else:
        total_records = len(live_predictions)
        total_machines = live_predictions["machineID"].nunique()
        
        
        live_predictions["risk_level"] = live_predictions[
            "failure_probability"
        ].apply(
            lambda p: (
                "HIGH"
                if p >= high_threshold
                else "MEDIUM"
                if p >= medium_threshold
                else "LOW"
            )
        )

        high_count = int(
            (live_predictions["risk_level"] == "HIGH").sum()
        )
        medium_count = int(
            (live_predictions["risk_level"] == "MEDIUM").sum()
        )
        low_count = int(
            (live_predictions["risk_level"] == "LOW").sum()
        )

        
        # live_predictions["risk_level"] = live_predictions["failure_probability"].apply(
        # lambda p: ("HIGH" if p >= high_threshold else "MEDIUM" if p >= medium_threshold else "LOW")

        # high_count = int((live_predictions["risk_level"] == "HIGH").sum())
        # medium_count = int((live_predictions["risk_level"] == "MEDIUM").sum())
        # low_count = int((live_predictions["risk_level"] == "LOW").sum())

        # )
        # k1, k2, k3, k4 = st.columns(4)
        # k1.metric("Sensor Records Scored", f"{total_records:,}")
        # k2.metric("Machines Represented", f"{total_machines:,}")
        # k3.metric("High-Risk Records", f"{high_count:,}")
        # k4.metric("Medium-Risk Records", f"{medium_count:,}")

        st.caption(
            "These results are predictions from the saved Kafka sensor replay. "
            "They do not represent continuous live factory monitoring."
        )

        risk_counts = pd.DataFrame({
            "Risk Level": ["HIGH", "MEDIUM", "LOW"],
            "Records": [high_count, medium_count, low_count]
        })

        left, right = st.columns(2)

        with left:
            fig = px.pie(
                risk_counts,
                names="Risk Level",
                values="Records",
                title="Predicted Risk Distribution",
                hole=0.5,
                color="Risk Level",
                color_discrete_map={
                    "HIGH": "#E45756",
                    "MEDIUM": "#F2A541",
                    "LOW": "#3A9D78"
                }
            )
            st.plotly_chart(fig, width="stretch")

        with right:
            st.markdown("**Highest-Risk Sensor Records**")

            top_risks = live_predictions.sort_values(
                "failure_probability", ascending=False
            ).head(10).copy()

            top_risks["failure_probability"] = (
                top_risks["failure_probability"] * 100
            ).round(2)

            st.dataframe(
                top_risks[
                    [
                        "machineID",
                        "datetime",
                        "predicted_failure",
                        "failure_probability",
                        "risk_level"
                    ]
                ].rename(columns={
                    "machineID": "Machine ID",
                    "datetime": "Timestamp",
                    "predicted_failure": "Predicted Failure",
                    "failure_probability": "Failure Probability (%)",
                    "risk_level": "Risk Level"
                }),
                width="stretch",
                hide_index=True
            )

        st.markdown("**All Scored Sensor Records**")

        display_live = live_predictions.sort_values(
            "failure_probability", ascending=False
        ).copy()

        display_live["failure_probability"] = (
            display_live["failure_probability"] * 100
        ).round(2)

        st.dataframe(
            display_live.rename(columns={
                "machineID": "Machine ID",
                "datetime": "Timestamp",
                "volt": "Voltage",
                "rotate": "Rotation",
                "pressure": "Pressure",
                "vibration": "Vibration",
                "predicted_failure": "Predicted Failure",
                "failure_probability": "Failure Probability (%)",
                "risk_level": "Risk Level"
            }),
            width="stretch",
            hide_index=True
        )

        st.download_button(
            "Download Kafka Predictions CSV",
            data=live_predictions.to_csv(index=False).encode("utf-8"),
            file_name="kafka_maintenance_predictions.csv",
            mime="text/csv"
        )

with tab_model:
    st.subheader("Model Evaluation")

    metrics_path = RESULTS_DIR / "model_metrics.txt"

    if metrics_path.exists():
        metrics_text = metrics_path.read_text(encoding="utf-8")
        st.code(metrics_text, language="text")
    else:
        st.warning("The model_metrics.txt file was not found.")

    actual = predictions["failure_within_24h"]
    predicted = predictions["prediction"]

    true_positive = int(((actual == 1) & (predicted == 1)).sum())
    true_negative = int(((actual == 0) & (predicted == 0)).sum())
    false_positive = int(((actual == 0) & (predicted == 1)).sum())
    false_negative = int(((actual == 1) & (predicted == 0)).sum())

    st.markdown("**Test-set confusion counts**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Positives", f"{true_positive:,}")
    c2.metric("True Negatives", f"{true_negative:,}")
    c3.metric("False Positives", f"{false_positive:,}")
    c4.metric("False Negatives", f"{false_negative:,}")

    st.caption(
        "Because machine failures are relatively rare, accuracy alone is not enough "
        "to assess the model. Recall, precision, and PR-AUC are also important."
    )

st.divider()
st.caption(
    "Manufacturing Predictive Maintenance | Initial dashboard using historical "
    "dataset predictions. Streaming integration is a future project phase."
)