from flask import Flask, render_template, redirect, url_for, flash, session, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask_mysqldb import MySQL
import os
from pymongo import MongoClient
from datetime import datetime
from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import SQLDatabase
from flask import jsonify
from langchain_core.prompts import ChatPromptTemplate
from flask import send_from_directory
import time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from flask import session
from sqlalchemy import create_engine, text


app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')

# MySQL configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '' 
app.config['MYSQL_DB'] = 'studentdb'

mysql = MySQL(app)

# Database configuration (hardcoded)
DATABASE_CONFIG = {
    "username": "root",
    "password": "",
    "host": "localhost",
    "port": 3306,
    "database": "studentdb"
}

def connect_database():
    """Connect to the database using the predefined configuration."""
    try:
        config = DATABASE_CONFIG
        # Create SQLAlchemy engine
        engine = create_engine(
            f"mysql+mysqlconnector://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
        )
        return SQLDatabase(engine)
    except Exception as e:
        print(f"Limited user database connection error: {e}")
        raise

# MongoDB configuration
mongo_client = MongoClient("mongodb://localhost:27017/")
chat_db = mongo_client["chat_history_db"]
chat_collection = chat_db["chats"]

# Database configuration for limited user
LIMITED_USER_CONFIG = {
    "username": "limited_user",  # Replace with your limited user credentials
    "password": "",  # Replace with your limited user password
    "host": "localhost",
    "port": 3306,
    "database": "studentdb"
}



def connect_limited_user():
    """Connect to the database using limited user credentials."""
    try:
        config = LIMITED_USER_CONFIG
        # Create SQLAlchemy engine
        engine = create_engine(
            f"mysql+mysqlconnector://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
        )
        return SQLDatabase(engine)
    except Exception as e:
        print(f"Limited user database connection error: {e}")
        raise

# Ensure indexes on portal_id and timestamp for optimization
chat_collection.create_index([("portal_id", 1), ("timestamp", -1)])

# LangChain LLM configuration
llm = ChatOllama(model="llama3.2")

db = None

def get_database_schema():
    """Get database schema using limited user credentials."""
    try:
        db = connect_limited_user()
        return db.get_table_info() if db else "Database is not connected."
    except Exception as e:
        print(f"Error fetching database schema: {e}")
        return "Error fetching database schema."

def execute_read_query(query):
    """Execute read-only queries using limited user credentials."""
    try:
        config = LIMITED_USER_CONFIG
        engine = create_engine(
            f"mysql+mysqlconnector://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
        )
        
        with engine.connect() as connection:
            result = connection.execute(text(query))
            columns = result.keys()
            results = result.fetchall()
            return columns, results
    except Exception as e:
        print(f"Error executing query: {e}")
        raise

def get_query_from_llm(question):
    try:
        schema = get_database_schema()
        template = """below is the schema of MYSQL database, read the schema carefully about the table and column names. Also take care of table or column name case sensitivity.
        Finally answer user's question in the form of SQL query.

        the schema of table :
        {schema}

        1.All the questions are only asked from studentsperformance table
        2.if 'top' or 'top students' is asked then give the student who scored most in that specific subject with that subject mark
        3.if 'details' is asken then give their respective CTPS,L1,L2,L3,L4,L5,PDS marks
        4. there is no T1 or T2 and no need to join anything

        Please only provide the SQL query and nothing else.

        also if the question asks name or NAME or Name it should pickup 'Name' column of database
        don't retrive portal_id only always retrive Name if portal_id is asked

        Example:
        question: how many subjects do we have in the database?
        SQL query: SELECT COUNT(*) FROM completion_status

        your turn:
        question: {question}
        SQL query:"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | ChatOllama(model="llama3.2")
        response = chain.invoke({"question": question, "schema": schema})
        return response.content
    except Exception as e:
        print(f"Error generating query: {e}")
        raise

def get_scores():
    """Fetch scores from the database or return default values."""
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT CTPS, L1, L2, L3, L4, L5, PDS FROM completion_status LIMIT 1")
        result = cur.fetchone()
        cur.close()
        if result:
            return {
                'CTPS': result[0],
                'L1': result[1],
                'L2': result[2],
                'L3': result[3],
                'L4': result[4],
                'L5': result[5],
                'PDS': result[6],
            }
    except Exception as e:
        print(f"Error fetching scores: {e}")
    return {'CTPS': 0, 'L1': 0, 'L2': 0, 'L3': 0, 'L4': 0, 'L5': 0, 'PDS': 0}

def dataframe_to_image(df, image_path='static/df_image.png'):
    """
    Converts a Pandas DataFrame into an image and saves it to a file with a unique name.

    Parameters:
    df (DataFrame): The Pandas DataFrame to render as an image.
    image_path (str): Path to save the image file.

    Returns:
    str: Path to the saved image.
    """
    try:
        # Generate a unique filename by appending a timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_path = f"static/df_image_{timestamp}.png"  # Include timestamp for uniqueness

        # Set plot style
        plt.style.use("seaborn-v0_8-whitegrid")
        
        # Calculate figure size based on DataFrame dimensions
        rows, cols = df.shape
        figsize = (min(12, max(8, cols)), min(8, max(4, rows * 0.5)))

        # Create a matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('tight')
        ax.axis('off')

        # Create the table with custom styling
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f0f0f0'] * len(df.columns),
            cellColours=[[('#ffffff' if i % 2 == 0 else '#f9f9f9') 
                         for j in range(len(df.columns))] 
                        for i in range(len(df.values))]
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        
        # Adjust column widths based on content
        for i, col in enumerate(df.columns):
            max_length = max(
                len(str(col)),
                *[len(str(val)) for val in df.iloc[:, i]]
            )
            table.auto_set_column_width([i])

        # Save the figure
        plt.savefig(image_path, 
                   bbox_inches='tight',
                   dpi=300,
                   facecolor='white',
                   edgecolor='none',
                   pad_inches=0.2)
        plt.close(fig)

        return image_path

    except Exception as e:
        print(f"Error generating image from DataFrame: {e}")
        plt.close('all')  # Ensure all figures are closed in case of error
        raise

    

@app.before_request
def check_session():
    if request.endpoint and 'static' not in request.endpoint:
        if 'portal_id' not in session:
            print("No portal_id in session")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        portal_id = request.form['portal_id']
        password = request.form['password']

        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT is_admin FROM users WHERE portal_id = %s AND password = %s", (portal_id, password))
            user = cur.fetchone()
            cur.close()
            if user:
                # Set session with debug print
                session['portal_id'] = portal_id
                print(f"Setting session for portal_id: {portal_id}")
                
                is_admin = user[0]
                if is_admin == 1:
                    flash(f"Welcome Admin, {portal_id}!", 'success')
                    return redirect(url_for("admin_home"))
                else:
                    flash(f"Welcome, {portal_id}!", 'success')
                    return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password!', 'danger')
        except Exception as e:
            print(f"Error during login: {e}")
            flash('An error occurred. Please try again.', 'danger')

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    portal_id = session.get('portal_id')  # Retrieve logged-in student's portal ID from the session
    if not portal_id:
        flash("You must log in first!", "danger")
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()

    # Fetch student data based on portal ID
    cur.execute("""
        SELECT Name, CTPS, L1, L2, L3, L4, L5, PDS, Total_score
        FROM studentsperformance
        WHERE Portal_ID = %s
    """, (portal_id,))
    student_data = cur.fetchone()  # Fetch one row
    cur.close()

    # Check if the student exists
    if not student_data:
        flash("No data found for your account. Please contact the administrator.", "warning")
        return redirect(url_for('login'))

    # Pass the data to the HTML template
    return render_template('studentDB.html', student=student_data)

@app.route('/admin_home')
def admin_home():
    # Check if user is logged in
    if 'portal_id' not in session:
        flash('Please login first.', 'warning')
        return redirect(url_for('login'))
    
    # Check if user is admin
    cur = mysql.connection.cursor()
    cur.execute("SELECT is_admin FROM users WHERE portal_id = %s", (session['portal_id'],))
    user = cur.fetchone()
    cur.close()
    
    if not user or user[0] != 1:
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('login'))
        
    return render_template('admin_home.html')

@app.route('/completion_status', methods=['GET', 'POST'])
def completion_status():
    """Display and update the completion scores for admin."""
    try:
        # Debug print for session
        print("Current session:", session)
        
        if 'portal_id' not in session:
            print("No portal_id in session")
            flash("You need to log in first.", "warning")
            return redirect(url_for('login'))

        # Check if user is admin with debug prints
        cur = mysql.connection.cursor()
        cur.execute("SELECT is_admin FROM users WHERE portal_id = %s", (session['portal_id'],))
        user = cur.fetchone()
        print(f"User admin check result: {user}")
        cur.close()

        if not user:
            print("No user found")
            flash("User not found!", "danger")
            return redirect(url_for('login'))
            
        if user[0] != 1:
            print("User is not admin")
            flash("Unauthorized access! Admin privileges required.", "danger")
            return redirect(url_for('dashboard'))

        if request.method == 'POST':
            # Extract scores from the form
            CTPS = request.form.get('CTPS')
            L1 = request.form.get('L1')
            L2 = request.form.get('L2')
            L3 = request.form.get('L3')
            L4 = request.form.get('L4')
            L5 = request.form.get('L5')
            PDS = request.form.get('PDS')

            # Update the `completion_status` table
            query = """
                UPDATE completion_status
                SET CTPS = %s, L1 = %s, L2 = %s, L3 = %s, L4 = %s, L5 = %s, PDS = %s, last_edited = NOW()
                WHERE id = 1
            """
            cur = mysql.connection.cursor()
            cur.execute(query, (CTPS, L1, L2, L3, L4, L5, PDS))
            mysql.connection.commit()

            # Update the `completed_` columns in `studentperformance`
            update_query = """
                UPDATE studentsperformance
                SET completed_CTPS = CASE WHEN CTPS >= %s THEN 1 ELSE 0 END,
                    completed_L1 = CASE WHEN L1 >= %s THEN 1 ELSE 0 END,
                    completed_L2 = CASE WHEN L2 >= %s THEN 1 ELSE 0 END,
                    completed_L3 = CASE WHEN L3 >= %s THEN 1 ELSE 0 END,
                    completed_L4 = CASE WHEN L4 >= %s THEN 1 ELSE 0 END,
                    completed_L5 = CASE WHEN L5 >= %s THEN 1 ELSE 0 END,
                    completed_PDS = CASE WHEN PDS >= %s THEN 1 ELSE 0 END
            """
            cur.execute(update_query, (CTPS, L1, L2, L3, L4, L5, PDS))
            mysql.connection.commit()
            cur.close()

            flash("Completion scores updated successfully!", "success")
            return redirect(url_for('completion_status'))

        # Fetch current scores for GET request
        scores = get_scores()
        return render_template('completion_status.html', scores=scores)
    except Exception as e:
        print(f"Error in completion_status: {e}")
        flash("An error occurred. Please try again later.", "danger")
        return redirect(url_for('login'))

@app.route('/llm', methods=['GET', 'POST'])
def chat():
    if db is None:
        try:
            connect_limited_user()
        except Exception as e:
            print(f"Database connection error: {e}")
            return jsonify({"error": f"Failed to connect to the database: {e}"}), 500

    if request.method == 'POST':
        try:
            question = request.form['question']
            query = get_query_from_llm(question)
            print(f"Generated query: {query}")

            # Execute query and get results
            columns, results = execute_read_query(query)
            
            # Convert results to DataFrame and generate image
            df = pd.DataFrame(results, columns=columns)
            image_path = dataframe_to_image(df)

            chat_id = str(datetime.now().timestamp())
            
            # Save the chat history to MongoDB
            chat_entry = {
                "portal_id": session.get('portal_id'),
                "chat_id": chat_id,
                "question": question,
                "query": query,
                "response": f"Query results:\n```sql\n{query}\n```",
                "image_path": image_path,
                "timestamp": datetime.now()
            }
            
            try:
                chat_collection.insert_one(chat_entry)
            except Exception as e:
                print(f"Error inserting chat entry: {e}")
                return jsonify({"error": "Failed to save chat entry"}), 500

            return jsonify({
                "success": True,
                "chat_id": chat_id,
                "question": question,
                "query": query,
                "has_image": bool(image_path),
                "image_path": f"/{image_path}" if image_path else None
            })

        except Exception as e:
            print(f"Error in chat route: {e}")
            return jsonify({"error": str(e)}), 500

    # For GET requests, return empty JSON instead of HTML template
    return render_template('gpt.html')

@app.route('/history', methods=['GET'])
def history():
    portal_id = session.get('portal_id')
    if not portal_id:
        flash("Please log in to view chat history.", "warning")
        return redirect(url_for('login'))

    try:
        # Strict portal_id matching for current user only
        current_user_chats = list(chat_collection.find(
            {"portal_id": {"$eq": portal_id}},  # Exact match only
            {
                "_id": 0,
                "chat_id": 1,
                "question": 1,
                "response": 1,
                "image_path": 1,
                "timestamp": 1,
                "portal_id": 1
            }
        ).sort("timestamp", -1))
        
        # Double check portal_id matches
        chats = [
            chat for chat in current_user_chats 
            if chat.get('portal_id') == portal_id
        ]
        
        # Clean up response data
        for chat in chats:
            chat.pop('portal_id', None)  # Remove portal_id before sending
            if isinstance(chat.get('timestamp'), datetime):
                chat["timestamp"] = chat["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            
        return render_template('history.html', chats=chats)
    except Exception as e:
        print(f"Error fetching chat history: {e}")
        flash("Failed to load chat history.", "danger")
        return redirect(url_for('dashboard'))

@app.route('/api/chats', methods=['GET'])
def get_chats():
    try:
        portal_id = session.get('portal_id')
        if not portal_id:
            return jsonify({"error": "Unauthorized access"}), 401

        # Strict portal_id matching pipeline
        pipeline = [
            {'$match': {'portal_id': {"$eq": portal_id}}},  # Exact match only
            {'$sort': {'timestamp': -1}},
            {'$group': {
                '_id': '$chat_id',
                'portal_id': {'$first': '$portal_id'},
                'chat_id': {'$first': '$chat_id'},
                'timestamp': {'$first': '$timestamp'},
                'last_message': {'$first': '$question'},
                'image_path': {'$first': '$image_path'}
            }},
            {'$match': {'portal_id': {"$eq": portal_id}}},  # Double check
            {'$project': {
                '_id': 0,
                'chat_id': 1,
                'timestamp': 1,
                'last_message': 1,
                'image_path': 1
            }}
        ]
        
        chats = list(chat_collection.aggregate(pipeline))
        
        # Format timestamps
        for chat in chats:
            if isinstance(chat.get('timestamp'), datetime):
                chat['timestamp'] = chat['timestamp'].isoformat()
        
        return jsonify(chats)
        
    except Exception as e:
        print(f"Error retrieving chats: {str(e)}")
        return jsonify({'error': 'Failed to retrieve chat history'}), 500

@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat_messages(chat_id):
    try:
        portal_id = session.get('portal_id')
        if not portal_id:
            return jsonify({"error": "Unauthorized"}), 401

        # Strict portal_id and chat_id matching
        messages = list(chat_collection.find(
            {
                'chat_id': chat_id,
                'portal_id': portal_id # Exact match only
            },
            {
                '_id': 0,
                'question': 1,
                'response': 1,
                'image_path': 1,
                'timestamp': 1
            }
        ).sort('timestamp', 1))
        
        if not messages:
            return jsonify({"error": "Chat not found or unauthorized"}), 404
            
        # Format timestamps
        for message in messages:
            if isinstance(message.get('timestamp'), datetime):
                message['timestamp'] = message['timestamp'].isoformat()
        
        return jsonify(messages)
        
    except Exception as e:
        print(f"Error in get_chat_messages: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def process_chat():
    if 'portal_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400
            
        message = data.get('message')
        if not message:
            return jsonify({"error": "Message is required"}), 400
            
        chat_id = data.get('chat_id') or f"chat_{int(time.time()*1000)}"
        
        # Generate and execute query
        query = get_query_from_llm(message)
        columns, results = execute_read_query(query)
        
        # Convert results to DataFrame and generate image
        df = pd.DataFrame(results, columns=columns)
        image_path = dataframe_to_image(df)
        
        # Prepare response
        response = f"Query results:\n```sql\n{query}\n```"
        
        # Save chat entry
        chat_entry = {
            "portal_id": session['portal_id'],
            "chat_id": chat_id,
            "question": message,
            "query": query,
            "response": response,
            "image_path": image_path,
            "timestamp": datetime.now()
        }
        
        chat_collection.insert_one(chat_entry)
        
        return jsonify({
            "reply": response,
            "has_image": bool(image_path),
            "image_path": f"/{image_path}" if image_path else None,
            "chat_id": chat_id
        })
            
    except Exception as e:
        print(f"Error processing chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/logout')
def logout():
    session.pop('portal_id', None)
    flash("Logged out successfully.", "info")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
