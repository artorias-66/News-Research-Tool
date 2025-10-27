# 🧠 Equity Research Tool

A **Streamlit-powered news research application** that allows you to input multiple news article URLs, automatically extract and clean their content, generate **local embeddings** using `sentence-transformers`, and answer questions about the articles — **all without any paid APIs**.

---

## 🚀 Features

- 📰 **Fetch and process multiple news URLs**
- 🧹 **Cleans text automatically** to remove noise (ads, tags, comments, etc.)
- 🧩 **Splits large articles into manageable chunks**
- 🧠 **Generates embeddings locally** using `all-MiniLM-L6-v2`
- 🔍 **Finds the most relevant sections** using cosine similarity
- 💬 **Ask natural-language questions** and get summarized answers
- ✅ **No API keys or internet AI models required**

---

## 🏗️ Tech Stack

- **Python 3.11+**
- **Streamlit**
- **Sentence Transformers**
- **scikit-learn**
- **LangChain Community**
- **dotenv**

---

## 📦 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/equity-research-tool.git
   cd equity-research-tool
Create and activate a virtual environment

bash
Copy code
python -m venv venv
# Activate it
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app

bash
Copy code
streamlit run main.py
⚙️ Usage
Enter up to 3 news article URLs in the sidebar.

Click “Process URLs” — the tool will fetch, clean, chunk, and embed the content locally.

Once processed, type any question about the articles in the input box.

The app retrieves and displays the most relevant excerpts, along with source links.

🧾 Project Structure
bash
Copy code
Equity Research Tool/
│
├── main.py                 # Main Streamlit app
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not pushed to GitHub)
├── embeddings_store.pkl    # Local embedding storage (auto-generated)
├── venv_311/               # Virtual environment (ignored)
├── venv_clean/             # (optional second venv, ignored)
└── .gitignore              # Ignore unnecessary files
🔒 Environment Variables
The app uses a .env file, but no API keys are required for this version.
If you wish to add your own keys (e.g., for LangChain integrations), create a .env file like:

ini
Copy code
OPENAI_API_KEY=your_api_key_here
💡 Example
After processing a few URLs, you can ask:

“What are the key market trends mentioned across these articles?”

and the tool will retrieve the most relevant text snippets with source links.

🧑‍💻 Author
Anubhav Verma
B.Tech IT | Backend Developer | AI Applications Engineer
🔗 LinkedIn

🪶 License
This project is open-source and available under the MIT License.

Made with ❤️ using Streamlit, LangChain, and Sentence Transformers.
