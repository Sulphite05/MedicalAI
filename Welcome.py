import streamlit as st


st.set_page_config(
    page_title="Welcome",
    page_icon="👋"
)

col1, col2 = st.columns([0.8, 5])

with col1: st.image('images/logo.png', width=100)
with col2: st.title("Welcome to NIKUD.ai")
st.divider()

st.subheader("Who We Are 💡")
st.markdown("""
NIKUD Research Hospital is a not-for-profit healthcare committed to delivering high-quality, affordable medical care. Registered with the Pakistan Society of Philanthropy, we provide accessible treatment for all, supported by fundraising initiatives for underserved patients.

Our specialties include Nephrology, Urology, Internal Medicine, Cardiology, Paediatrics, and Oncology, backed by a state-of-the-art laboratory and radiology services. Pioneering safety, our Dialysis Program features dedicated machines for Hepatitis B/C-positive patients.

With urgent care available during business hours, NIKUD combines advanced infrastructure with compassionate service—because every life deserves elite care.
""")

st.subheader("Our App Features 🏆")
st.write("""
    1. **AI-based Retinopathy Detection**: Upload the x-ray of your retina to know the severity of diabetes.\n
""")

st.image('images/building.jpg', caption='NIKUD')

footer = """
    <style>
        .footer {
            bottom: 0;
            left: 0;
            right: 0;
            text-align: center;
            padding: 10px;
            font-size: 12px;
            color: #555;
        }
    </style>
    <div class="footer">
        <p>Made with lots of ☕ This is a demo project.</p>
    </div>
"""

st.markdown(footer, unsafe_allow_html=True)
