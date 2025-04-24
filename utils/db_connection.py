import pyodbc

def get_connection():
    conn_str = (
        "DRIVER={SQL Server};"
        "SERVER=MSISWORDTHINUKA;"
        "DATABASE=saxo_db;"
        "Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str)
