import argparse
import nltk_summarizer
import converter
import transformers_summarizer
import sumy_summarizer
import streamlit as st
import clipboard


def main():
    output = ""
    with st.sidebar:
        logo_url = 'penguuuuu.png'
        st.sidebar.image(logo_url)
        st.header("Summary Bot")
        pdf = st.file_uploader("Upload pdf")

        method = st.selectbox(
            "Choose a method to summarize", ["sumy", "nltk", "pegasus"],
            index=None,
            placeholder="Select method...",
        )
        if st.button("Summarize"):
            with st.spinner("In progress..."):
                if method == "nltk":
                    #summarizer_text = converter.PDFToTextConverter(pdf.name)
                    #output = summarizer_text.disp()
                    summarizer_nltk = nltk_summarizer.PDFSummarizer(pdf.name)
                    summarizer_nltk.summarize()
                    output = summarizer_nltk.suma()
                elif method == "pegasus":
                    summarizer_pegasus = transformers_summarizer.PDFSummarizer(pdf.name)
                    summarizer_pegasus.summarize()
                    output = summarizer_pegasus.suma()
                elif method == "sumy":
                    summarizer_sumy = sumy_summarizer.PDFSummarizer(pdf.name)
                    summarizer_sumy.summarize()
                    output = summarizer_sumy.suma()

    with st.container():
        st.write("Your summarization is here!")
        st.info(output)



if __name__ == "__main__":
    main()
