# MySQL database connection 
import mysql.connector 
 
def connect_db(): 
    try: 
        conn = mysql.connector.connect( 
            host='localhost', 
            user='your_username', 
            password='your_password', 
            database='handwritten_text_db' 
        ) 
        return conn 
    except mysql.connector.Error as e: 
        print(f'Error connecting to MySQL: {e}') 
        return None 
