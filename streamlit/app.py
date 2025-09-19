import streamlit as st
from prediction import predict
# Title
st.title(" Sentiment Analysis on IMDB Reviews")

# Text input
review = st.text_area("Enter an IMDB Review", height=200, placeholder="Type your movie review here...")

# Submit button
if st.button("Analyze Sentiment"):
    if review.strip():
        st.subheader("Prediction")
        prediction = predict(review)
        result = prediction[0]
        proba = prediction[1]
        if result == "Positive Review":
            st.success(f"{result}")
        else:
            st.error(f"{result}")

        st.info(f"Probability: {proba*100:.2f}%")
    else:
        st.warning("Please enter a review before analyzing.")

# Footer
st.markdown("---")
st.caption("Model can make mistake")
