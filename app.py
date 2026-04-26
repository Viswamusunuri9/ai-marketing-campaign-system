import streamlit as st
from src.predict import predict_customer

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Marketing Campaign Intelligence System")

st.title("📢 Marketing Campaign Intelligence System")
st.markdown("Predict which customers should be targeted for campaigns.")

# ----------------------------
# CUSTOMER PROFILE
# ----------------------------
st.subheader("👤 Customer Profile")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Income", min_value=0, value=50000)
    kidhome = st.number_input("Kids at home", min_value=0, value=0)
    teenhome = st.number_input("Teens at home", min_value=0, value=0)

with col2:
    year_birth = st.number_input("Year of Birth", min_value=1940, max_value=2024, value=1990)

# ----------------------------
# SPENDING
# ----------------------------
st.subheader("💰 Spending Behavior")

col3, col4 = st.columns(2)

with col3:
    mnt_wines = st.number_input("Wine Spending", min_value=0, value=100)
    mnt_meat = st.number_input("Meat Spending", min_value=0, value=100)
    mnt_fruits = st.number_input("Fruit Spending", min_value=0, value=20)

with col4:
    mnt_fish = st.number_input("Fish Spending", min_value=0, value=20)
    mnt_sweets = st.number_input("Sweet Spending", min_value=0, value=20)
    mnt_gold = st.number_input("Gold Products Spending", min_value=0, value=20)

# ----------------------------
# BEHAVIOR (🔥 IMPORTANT)
# ----------------------------
st.subheader("🛒 Purchase Behavior")

col5, col6 = st.columns(2)

with col5:
    web_purchases = st.number_input("Web Purchases", min_value=0, value=2)
    store_purchases = st.number_input("Store Purchases", min_value=0, value=2)

with col6:
    web_visits = st.number_input("Web Visits per Month", min_value=0, value=5)

# ----------------------------
# CAMPAIGN HISTORY
# ----------------------------
st.subheader("📢 Campaign History")

col7, col8, col9 = st.columns(3)

with col7:
    acc1 = st.selectbox("Campaign 1", [0, 1])
    acc2 = st.selectbox("Campaign 2", [0, 1])

with col8:
    acc3 = st.selectbox("Campaign 3", [0, 1])
    acc4 = st.selectbox("Campaign 4", [0, 1])

with col9:
    acc5 = st.selectbox("Campaign 5", [0, 1])

# ----------------------------
# PREDICT
# ----------------------------
if st.button("🚀 Predict Campaign Response"):

    input_data = {
        # Profile
        "Income": income,
        "Kidhome": kidhome,
        "Teenhome": teenhome,
        "Year_Birth": year_birth,

        # Spending
        "MntWines": mnt_wines,
        "MntMeatProducts": mnt_meat,
        "MntFruits": mnt_fruits,
        "MntFishProducts": mnt_fish,
        "MntSweetProducts": mnt_sweets,
        "MntGoldProds": mnt_gold,

        # Behavior
        "NumWebPurchases": web_purchases,
        "NumCatalogPurchases": 0,  # optional default
        "NumStorePurchases": store_purchases,
        "NumWebVisitsMonth": web_visits,

        # Campaign history
        "AcceptedCmp1": acc1,
        "AcceptedCmp2": acc2,
        "AcceptedCmp3": acc3,
        "AcceptedCmp4": acc4,
        "AcceptedCmp5": acc5,
    }

    with st.spinner("Analyzing customer behavior..."):
        prob = predict_customer(input_data)

    # ----------------------------
    # OUTPUT
    # ----------------------------
    st.subheader("📊 Prediction Result")

    if prob > 0.6:
        st.success(f"🎯 HIGH VALUE CUSTOMER ({prob:.2f})")
        st.write("Strong candidate for campaign targeting.")

    elif prob > 0.3:
        st.info(f"🟡 MEDIUM POTENTIAL ({prob:.2f})")
        st.write("Consider targeted or optimized campaign.")

    else:
        st.warning(f"⚠️ LOW RESPONSE PROBABILITY ({prob:.2f})")
        st.write("Not recommended for campaign spend.")
    st.progress(float(prob))
# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.caption("ML-powered Marketing Decision System")