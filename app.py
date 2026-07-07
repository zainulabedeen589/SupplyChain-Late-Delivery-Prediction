import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Supply Chain Late Delivery Predictor",
    page_icon="📦",
    layout="wide",
)

# ---------------- Animated background + floating text CSS ----------------
st.markdown(
    """
    <style>
    /* Hide default header background */
    .stApp > header {
        background-color: transparent;
    }

    /* Animated mesh gradient background */
    .stApp {
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #020617;
        background-image:
            radial-gradient(at 0% 0%, rgba(59,130,246,0.9) 0px, transparent 55%),
            radial-gradient(at 100% 0%, rgba(14,165,233,0.9) 0px, transparent 55%),
            radial-gradient(at 0% 100%, rgba(236,72,153,0.9) 0px, transparent 55%),
            radial-gradient(at 100% 100%, rgba(234,179,8,0.9) 0px, transparent 55%);
        background-size: 210% 210%;
        animation: meshMove 25s ease-in-out infinite;
    }

    @keyframes meshMove {
        0%   { background-position: 0% 0%; }
        25%  { background-position: 100% 0%; }
        50%  { background-position: 100% 100%; }
        75%  { background-position: 0% 100%; }
        100% { background-position: 0% 0%; }
    }

    /* Main content card */
    main .block-container {
        background-color: rgba(15,23,42,0.92);
        border-radius: 20px;
        padding: 1.8rem 2.2rem;
        box-shadow: 0 25px 50px rgba(0,0,0,0.7);
        border: 1px solid rgba(148,163,184,0.35);
        position: relative;
        overflow: hidden; /* for inner floating text */
    }

    /* Floating text INSIDE card, behind parameters */
    .inner-floating-name {
        position: absolute;
        inset: 0;
        pointer-events: none;
        z-index: 0;  /* behind widgets */
        display: flex;
        justify-content: center;
        align-items: center;
        opacity: 0.12;
        font-size: 4.2rem;
        letter-spacing: 0.35rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #e5e7eb;
        text-shadow:
            0 0 24px rgba(15,23,42,0.9),
            0 0 60px rgba(37,99,235,0.8);
        animation: floatInnerName 26s ease-in-out infinite;
    }

    @keyframes floatInnerName {
        0%   { transform: translate(-60px, -10px); }
        25%  { transform: translate(40px, -30px); }
        50%  { transform: translate(60px, 25px); }
        75%  { transform: translate(-40px, 30px); }
        100% { transform: translate(-60px, -10px); }
    }

    /* Make Streamlit widgets above floating text */
    .stMarkdown, .stText, .stCaption, p,
    .stButton, .stSlider, .stSelectbox, .stNumberInput, .stForm,
    .stTabs, .stMetric, .stAlert {
        position: relative;
        z-index: 1;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] > div:first-child {
        background: rgba(15,23,42,0.95);
        border-right: 1px solid rgba(148,163,184,0.35);
    }

    /* Headings & text colors */
    h1, h2, h3, h4, h5, h6 {
        color: #e5e7eb;
        position: relative;
        z-index: 1;
    }
    .stMarkdown, .stText, .stCaption, p {
        color: #cbd5f5;
    }

    /* Tabs styling (optional) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(15,23,42,0.85);
        padding: 10px 16px;
        border-radius: 999px;
        color: #e5e7eb;
        border: 1px solid rgba(148,163,184,0.4);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(59,130,246,0.95);
        border-color: rgba(96,165,250,1);
        color: #f9fafb;
    }

    @media (max-width: 768px) {
        .inner-floating-name {
            font-size: 2.6rem;
            letter-spacing: 0.22rem;
        }
        main .block-container {
            padding: 1.2rem 1.4rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Load model & mappings ----------------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("model/rf_late_delivery_model.pkl")
        mappings = joblib.load("model/freq_mappings.pkl")
        return model, mappings
    except Exception as e:
        st.error("Model assets could not be loaded.")
        st.caption("Please check model paths and file permissions.")
        st.caption(f"Technical details: {e}")
        return None, None


rf_model, freq_mappings = load_assets()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("📦 Late Delivery Risk")

    st.markdown(
        """
        Estimate the probability that an order will be delivered late,
        using a trained machine learning model.
        """
    )

    st.markdown("### Display")
    show_hints = st.toggle("Show field hints", value=True, key="show_hints")

    st.markdown("---")
    st.caption("Model: Random Forest classifier")
    st.caption("Use as a decision-support tool, not a single source of truth.")

    # ---------- Author section ----------
    st.markdown("---")
    st.markdown(
        """
        **Author**: ZAINUL ABEDEEN  

        [GitHub](https://github.com/zainulabedeen589)
        [LinkedIn](https://www.linkedin.com/in/zainulabedeen589/)
        """,
        unsafe_allow_html=True,
    )

# ---------------- Main header ----------------
st.markdown("### Supply Chain Intelligence")
st.title("Supply Chain Late Delivery Predictor")

st.markdown(
    """
    Provide the order details below to generate a late‑delivery risk prediction.  
    Built for planners, analysts and operations managers.
    """
)

st.divider()

# ---------------- Tabs ----------------
tab_predict, tab_insights = st.tabs(["🔮 Prediction", "📊 Insights"])

# ---------------- Prediction tab ----------------
with tab_predict:
    if rf_model and freq_mappings:
        # Ye div main card ke andar floating text ko inject karega
        st.markdown(
            """
            <div class="inner-floating-name">
                ZAINUL&nbsp;ABEDEEN
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Layout: form + info panel ---
        col_form, col_info = st.columns([2, 1])

        # ----- Form -----
        with col_form:
            with st.form("prediction_form", clear_on_submit=False):
                st.subheader("Order details")

                col1, col2, col3 = st.columns(3)

                with col1:
                    order_type = st.selectbox(
                        "Payment type",
                        options=list(freq_mappings["Type"].keys()),
                    )
                    scheduled_days = st.number_input(
                        "Scheduled shipping days",
                        min_value=0,
                        max_value=30,
                        value=4,
                        help=(
                            "Planned days between order and shipment."
                            if show_hints
                            else None
                        ),
                    )
                    cat_name = st.selectbox(
                        "Product category",
                        options=list(freq_mappings["Category Name"].keys()),
                    )

                with col2:
                    cust_segment = st.selectbox(
                        "Customer segment",
                        options=list(freq_mappings["Customer Segment"].keys()),
                    )
                    dept_name = st.selectbox(
                        "Department name",
                        options=list(freq_mappings["Department Name"].keys()),
                    )
                    order_region = st.selectbox(
                        "Order region",
                        options=list(freq_mappings["Order Region"].keys()),
                    )

                with col3:
                    shipping_mode = st.selectbox(
                        "Shipping mode",
                        options=list(freq_mappings["Shipping Mode"].keys()),
                    )
                    order_month = st.slider(
                        "Order month",
                        min_value=1,
                        max_value=12,
                        value=6,
                        help=(
                            "Calendar month of order placement." if show_hints else None
                        ),
                    )
                    order_hour = st.slider(
                        "Order hour (24h)",
                        min_value=0,
                        max_value=23,
                        value=12,
                        help=(
                            "Hour of the day the order was placed."
                            if show_hints
                            else None
                        ),
                    )

                st.markdown("---")
                submit_button = st.form_submit_button(
                    label="⚙️ Predict late delivery risk",
                    use_container_width=True,
                )

        # ----- Info panel -----
        with col_info:
            st.subheader("Model overview")
            st.markdown(
                """
                - Algorithm: Random Forest classifier  
                - Features: payment, product, customer, region, timing  
                - Output: probability that delivery will be **late**  
                """
            )
            st.caption(
                "Interpret predictions together with business context, "
                "capacity constraints and customer commitments."
            )

        # ----- Prediction -----
        if submit_button:
            with st.spinner("Running prediction..."):
                input_data = {
                    "Type": order_type,
                    "Days for shipment (scheduled)": scheduled_days,
                    "Category Name": cat_name,
                    "Customer Segment": cust_segment,
                    "Department Name": dept_name,
                    "Order Region": order_region,
                    "Shipping Mode": shipping_mode,
                    "order_month": order_month,
                    "order_hour": order_hour,
                }

                input_df = pd.DataFrame([input_data])

                cat_cols = [
                    "Type",
                    "Category Name",
                    "Customer Segment",
                    "Department Name",
                    "Order Region",
                    "Shipping Mode",
                ]

                for col in cat_cols:
                    input_df[f"{col}_freq"] = input_df[col].map(
                        lambda x: freq_mappings[col].get(x, 0)
                    )

                input_df = input_df.drop(columns=cat_cols)

                expected_cols = [
                    "Days for shipment (scheduled)",
                    "order_month",
                    "order_hour",
                    "Type_freq",
                    "Category Name_freq",
                    "Customer Segment_freq",
                    "Department Name_freq",
                    "Order Region_freq",
                    "Shipping Mode_freq",
                ]

                input_encoded = input_df[expected_cols]

                prediction = rf_model.predict(input_encoded)[0]
                probability = rf_model.predict_proba(input_encoded)[0][1]

            st.divider()
            st.subheader("Prediction result")

            prob_pct = probability * 100
            prob_text = f"{prob_pct:.1f}%"

            # --- KPI metric row ---
            kpi_col1, kpi_col2 = st.columns([1, 2])
            with kpi_col1:
                label = (
                    "Late delivery probability"
                    if prediction == 1
                    else "Late delivery risk"
                )
                st.metric(label=label, value=prob_text)

            with kpi_col2:
                st.caption(
                    "This value represents the model‑estimated probability "
                    "that the order will be delivered late."
                )

            # --- Result cards ---
            col_res, col_res_info = st.columns([1.6, 1])

            with col_res:
                if prediction == 1:
                    st.error("⚠️ High risk of late delivery")
                    st.markdown(
                        f"**Estimated probability of late delivery:** {prob_text}"
                    )
                    st.toast(
                        "Order flagged as high late‑delivery risk.",
                        icon="⚠️",
                    )
                else:
                    st.success("✅ Order likely to be delivered on time")
                    st.markdown(f"**Estimated risk of late delivery:** {prob_text}")
                    st.toast(
                        "Order assessed as low late‑delivery risk.",
                        icon="✅",
                    )
                    if prob_pct < 20:
                        st.balloons()

            with col_res_info:
                st.markdown("#### Suggested actions")
                if prediction == 1:
                    st.markdown(
                        """
                        - Review shipping mode and carrier capacity  
                        - Consider expediting or re‑prioritizing this order  
                        - Communicate proactively with the customer  
                        """
                    )
                else:
                    st.markdown(
                        """
                        - Current shipping plan appears adequate  
                        - Monitor high‑value or urgent orders  
                        - Use this as a reference for similar orders  
                        """
                    )

    else:
        st.warning(
            "Prediction model is not currently available. "
            "Please verify the model files or contact the administrator."
        )

# ---------------- Insights tab ----------------
with tab_insights:
    st.subheader("How to interpret the predictions")

    st.markdown(
        """
        - **0–20%**: Low late‑delivery risk, standard processing is usually sufficient.  
        - **20–50%**: Medium risk, monitor closely and consider contingency options.  
        - **50%+**: High risk, proactively adjust shipping plans and communicate with the customer.  
        """
    )
    st.caption(
        "You can extend this tab later to show historical distributions, feature importance, "
        "or recent orders once you connect data."
    )
