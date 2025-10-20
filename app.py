import streamlit as st
import torch
from model import load_model_and_tokenizer
import time
import random

# Page configuration
st.set_page_config(
    page_title="اردو چیٹ بوٹ - Urdu Chat Bot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .urdu-text {
        font-family: 'Noto Sans Arabic', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.1rem;
        line-height: 1.6;
        direction: rtl;
        text-align: right;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 12px 18px;
        border-radius: 18px;
        margin: 8px 0;
        border: 1px solid #90caf9;
        max-width: 70%;
        margin-left: auto;
        margin-right: 0;
        direction: rtl;
    }
    .bot-message {
        background-color: #f5f5f5;
        padding: 12px 18px;
        border-radius: 18px;
        margin: 8px 0;
        border: 1px solid #e0e0e0;
        max-width: 70%;
        margin-right: auto;
        margin-left: 0;
        direction: rtl;
    }
    .chat-container {
        height: 500px;
        overflow-y: auto;
        padding: 20px;
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        margin-bottom: 20px;
        background-color: #fafafa;
        direction: rtl;
    }
    .stButton button {
        border-radius: 20px;
        height: 2.8rem;
        font-size: 1rem;
    }
    .sample-question {
        background-color: #f0f8ff;
        border: 1px solid #b3d9ff;
        border-radius: 10px;
        padding: 8px 12px;
        margin: 5px;
        cursor: pointer;
        text-align: center;
        direction: rtl;
    }
    .sample-question:hover {
        background-color: #e6f2ff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'loading_error' not in st.session_state:
    st.session_state.loading_error = None
if 'auto_send' not in st.session_state:
    st.session_state.auto_send = False
if 'suggested_text' not in st.session_state:
    st.session_state.suggested_text = ""

def load_chatbot():
    """Load the chatbot model"""
    try:
        with st.spinner('🔄 اردو چیٹ بوٹ لوڈ ہو رہی ہے... براہ کرم انتظار کریں'):
            chatbot = load_model_and_tokenizer()
            st.session_state.chatbot = chatbot
            st.session_state.model_loaded = True
            st.session_state.loading_error = None
        return True
    except Exception as e:
        st.session_state.loading_error = str(e)
        return False

def generate_response(user_input):
    """Generate bot response"""
    if st.session_state.model_loaded and st.session_state.chatbot:
        try:
            # Add typing indicator
            with st.spinner("🔮 بوٹ سوچ رہا ہے..."):
                time.sleep(0.5)  # Simulate thinking time
                response = st.session_state.chatbot.generate_response(
                    user_input, 
                    max_length=60, 
                    temperature=0.7
                )
            return response
        except Exception as e:
            return f"معذرت، جواب بنانے میں مسئلہ ہوا: {str(e)}"
    else:
        return "ماڈل لوڈ نہیں ہوا۔ براہ کرم ریفریش کریں۔"

def display_chat_message(role, message):
    """Display a chat message"""
    if role == "user":
        st.markdown(f'<div class="user-message urdu-text"><b>آپ:</b> {message}</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message urdu-text"><b>بوٹ:</b> {message}</div>', 
                   unsafe_allow_html=True)

def handle_suggested_question(question):
    """Handle when a suggested question is clicked"""
    st.session_state.suggested_text = question
    st.session_state.auto_send = True
    st.rerun()

def process_auto_send():
    """Process auto-send when suggested question is clicked"""
    if st.session_state.auto_send and st.session_state.suggested_text:
        user_input = st.session_state.suggested_text
        # Add user message to chat history
        st.session_state.chat_history.append(("user", user_input))
        
        # Generate and add bot response
        bot_response = generate_response(user_input)
        st.session_state.chat_history.append(("bot", bot_response))
        
        # Reset auto-send flags
        st.session_state.auto_send = False
        st.session_state.suggested_text = ""
        
        # Rerun to show the new messages
        st.rerun()

def main():
    # Header
    st.markdown('<h1 class="main-header urdu-text">💬 اردو چیٹ بوٹ</h1>', 
               unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.markdown("### ⚙️ کنٹرولز")
        
        if st.button("🔄 ماڈل دوبارہ لوڈ کریں", use_container_width=True):
            if load_chatbot():
                st.success("✅ ماڈل کامیابی سے لوڈ ہو گیا!")
        
        if st.button("🗑️ بات چیت صاف کریں", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.auto_send = False
            st.session_state.suggested_text = ""
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ ہدایات")
        st.markdown("""
        <div class="urdu-text">
        - اردو میں لکھیں
        - عام گفتگو کے جملے استعمال کریں
        - مختصر اور واضح بات کریں
        - بوٹ کو جواب دینے کے لیے وقت دیں
        - نیچے دیے گئے سوالات پر کلک کریں
        </div>
        """, unsafe_allow_html=True)
        
        # Display model status
        st.markdown("---")
        st.markdown("### 📊 سسٹم کی حالت")
        if st.session_state.model_loaded:
            st.success("✅ ماڈل لوڈ ہو گیا")
        else:
            st.error("❌ ماڈل لوڈ نہیں ہوا")
    
    # Main chat area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Chat container
        st.markdown("### 💬 گفتگو")
        chat_container = st.container()
        
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Display chat history
            for role, message in st.session_state.chat_history:
                display_chat_message(role, message)
            
            # Display loading error if any
            if st.session_state.loading_error:
                st.error(f"لوڈنگ میں خرابی: {st.session_state.loading_error}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Input area
        st.markdown("---")
        col_input, col_send = st.columns([4, 1])
        
        with col_input:
            user_input = st.text_input(
                "اپنا پیغام یہاں لکھیں:",
                placeholder="اردو میں لکھیں...",
                key="user_input",
                label_visibility="collapsed",
                value=st.session_state.suggested_text if st.session_state.auto_send else ""
            )
        
        with col_send:
            send_button = st.button("📤 بھیجیں", use_container_width=True)
        
        # Handle send button
        if send_button and user_input.strip():
            # Add user message to chat history
            st.session_state.chat_history.append(("user", user_input.strip()))
            
            # Generate and add bot response
            bot_response = generate_response(user_input.strip())
            st.session_state.chat_history.append(("bot", bot_response))
            
            # Clear input and rerun
            st.session_state.suggested_text = ""
            st.rerun()
    
    # Load model on first run
    if not st.session_state.model_loaded and not st.session_state.loading_error:
        if load_chatbot():
            st.success("✅ اردو چیٹ بوٹ تیار ہے! اب آپ بات چیت شروع کر سکتے ہیں۔")
            
            # Add welcome message
            if not st.session_state.chat_history:
                welcome_messages = [
                    "ہیلو! میں اردو چیٹ بوٹ ہوں۔ آپ کیسے ہیں؟",
                    "السلام علیکم! میں آپ کی کس طرح مدد کر سکتا ہوں؟",
                    "ہیلو! آج آپ کیسے ہیں؟ میں آپ سے بات کرنے کے لیے تیار ہوں۔"
                ]
                welcome_msg = random.choice(welcome_messages)
                st.session_state.chat_history.append(("bot", welcome_msg))
                st.rerun()

    # Process auto-send if a suggested question was clicked
    if st.session_state.auto_send:
        process_auto_send()

    # Sample conversation starters
    st.markdown("---")
    st.markdown("### 💡 بات چیت شروع کرنے کے لیے تجاویز:")
    st.markdown('<div class="urdu-text">نیچے دیے گئے سوالات پر کلک کریں اور بوٹ کا جواب دیکھیں</div>', unsafe_allow_html=True)
    
    sample_questions = [
        "زندگی بہت مشکل لگ رہی ہے",
        "تم ناراض ہو",
        "کیا تم میری مدد کر سکتے ہو",
        "آج موسم کیسا ہے",
        "تم کہاں رہتے ہو",
        "تم نے یونیورسٹی کا نتیجہ آ گیا",
        "ہیلو کیا حال ہے",
        "تمہارا نام کیا ہے",
        "کیا آج سکول جاؤں گا",
        "مجھے اردو سیکھنی ہے"
    ]
    
    # Display sample questions in a grid with proper click handling
    cols = st.columns(3)
    for idx, question in enumerate(sample_questions):
        with cols[idx % 3]:
            # Use a unique key for each button
            if st.button(
                question, 
                key=f"sample_{idx}", 
                use_container_width=True,
                on_click=handle_suggested_question,
                args=(question,)
            ):
                # This will be handled by the callback function
                pass

if __name__ == "__main__":
    main()
