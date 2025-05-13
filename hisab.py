import sqlite3
import datetime

# Database setup
def setup_database():
    conn = sqlite3.connect('hisaab_db.sqlite')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        sno INTEGER PRIMARY KEY AUTOINCREMENT,
        id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL
    )
    ''')
    
    # Create transactions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        sno INTEGER PRIMARY KEY AUTOINCREMENT,
        giver_name TEXT NOT NULL,
        giver_id TEXT NOT NULL,
        receiver_name TEXT NOT NULL,
        receiver_id TEXT NOT NULL,
        amount REAL NOT NULL,
        date TEXT NOT NULL,
        note TEXT,
        FOREIGN KEY (giver_id) REFERENCES users(id),
        FOREIGN KEY (receiver_id) REFERENCES users(id)
    )
    ''')
    
    # Create balance table (to track balances between pairs of users)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS balances (
        sno INTEGER PRIMARY KEY AUTOINCREMENT,
        user1_id TEXT NOT NULL,
        user1_name TEXT NOT NULL,
        user2_id TEXT NOT NULL,
        user2_name TEXT NOT NULL,
        net_amount REAL NOT NULL DEFAULT 0,
        last_updated TEXT NOT NULL,
        UNIQUE(user1_id, user2_id)
    )
    ''')
    
    conn.commit()
    conn.close()

# User management functions
def add_user():
    conn = sqlite3.connect('hisaab_db.sqlite')
    cursor = conn.cursor()
    
    print("\n===== ADD NEW USER =====")
    user_id = input("Enter user ID (username/email): ")
    
    # Check if user already exists
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    if cursor.fetchone():
        print("User ID already exists. Please use a different ID.")
        conn.close()
        return
    
    name = input("Enter user name: ")
    
    cursor.execute("INSERT INTO users (id, name) VALUES (?, ?)", (user_id, name))
    conn.commit()
    
    print(f"User {name} ({user_id}) added successfully!")
    conn.close()

def get_all_users():
    conn = sqlite3.connect('hisaab_db.sqlite')
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users ORDER BY name")
    users = cursor.fetchall()
    
    conn.close()
    return users

def print_all_users():
    users = get_all_users()
    
    if not users:
        print("No users found in the system.")
        return
    
    print("\n===== ALL USERS =====")
    for user in users:
        print(f"[{user[0]}] {user[2]} (ID: {user[1]})")

def get_user_by_id(user_id):
    conn = sqlite3.connect('hisaab_db.sqlite')
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    
    conn.close()
    return user

# Transaction functions
def add_transaction():
    conn = sqlite3.connect('hisaab_db.sqlite')
    cursor = conn.cursor()
    
    print("\n===== NEW EXPENSE ENTRY =====")
    print("Choose transaction type:")
    print("1. Split between specific people")
    print("2. Direct payment (one person owes another)")
    print("3. Back to main menu")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        # Get all users for reference
        print_all_users()
        
        # Who paid
        payer_id = input("\nEnter the ID of the person who paid: ")
        payer = get_user_by_id(payer_id)
        if not payer:
            print("User not found!")
            conn.close()
            return
        
        # Amount paid
        amount = float(input("Enter the total amount paid: "))
        if amount <= 0:
            print("Amount must be greater than zero.")
            conn.close()
            return
        
        # Transaction note
        note = input("Enter a note for this transaction (e.g., 'Dinner', 'Movie'): ")
        
        # Include yourself?
        include_self = input("Include payer in the split? (y/n): ").lower() == 'y'
        
        # Get recipients
        print("\nEnter the IDs of people to split with (comma separated, e.g. 'user1,user2'):")
        recipients_input = input("Recipients: ")
        recipient_ids = [id.strip() for id in recipients_input.split(',')]
        
        if not include_self and payer_id in recipient_ids:
            recipient_ids.remove(payer_id)
        elif include_self and payer_id not in recipient_ids:
            recipient_ids.append(payer_id)
        
        # Validate all recipients exist
        valid_recipients = []
        for rid in recipient_ids:
            user = get_user_by_id(rid)
            if user:
                valid_recipients.append(user)
            else:
                print(f"Warning: User with ID '{rid}' not found and will be skipped.")
        
        if not valid_recipients:
            print("No valid recipients found. Transaction cancelled.")
            conn.close()
            return
        
        # Calculate split amount
        split_amount = amount / len(valid_recipients)
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Record transactions
        for recipient in valid_recipients:
            if recipient[1] != payer_id:  # Skip if recipient is the payer
                # Add transaction record
                cursor.execute("""
                INSERT INTO transactions (giver_name, giver_id, receiver_name, receiver_id, amount, date, note)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (recipient[2], recipient[1], payer[2], payer[1], split_amount, current_date, note))
                
                # Update balance (using sorted user IDs to ensure consistency)
                user_ids = sorted([payer[1], recipient[1]])
                user_names = [get_user_by_id(user_ids[0])[2], get_user_by_id(user_ids[1])[2]]
                
                # Check if balance record exists
                cursor.execute("""
                SELECT * FROM balances WHERE (user1_id = ? AND user2_id = ?)
                """, (user_ids[0], user_ids[1]))
                
                balance = cursor.fetchone()
                
                if balance:
                    # Update existing balance
                    net_amount = balance[5]
                    if payer[1] == user_ids[0]:
                        net_amount += split_amount
                    else:
                        net_amount -= split_amount
                    
                    cursor.execute("""
                    UPDATE balances SET net_amount = ?, last_updated = ?
                    WHERE user1_id = ? AND user2_id = ?
                    """, (net_amount, current_date, user_ids[0], user_ids[1]))
                else:
                    # Create new balance record
                    net_amount = split_amount if payer[1] == user_ids[0] else -split_amount
                    
                    cursor.execute("""
                    INSERT INTO balances (user1_id, user1_name, user2_id, user2_name, net_amount, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """, (user_ids[0], user_names[0], user_ids[1], user_names[1], net_amount, current_date))
        
        conn.commit()
        print(f"\nExpense of ₹{amount:.2f} recorded successfully!")
        print(f"Note: {note}")
        if valid_recipients and len(valid_recipients) > 1:
            print(f"Split amount: ₹{split_amount:.2f} per person")
    
    elif choice == '2':
        # Direct payment (one person owes another)
        print_all_users()
        
        # Who is owed (creditor)
        creditor_id = input("\nEnter the ID of the person who is owed money: ")
        creditor = get_user_by_id(creditor_id)
        if not creditor:
            print("User not found!")
            conn.close()
            return
        
        # Who owes (debtor)
        debtor_id = input("Enter the ID of the person who owes money: ")
        debtor = get_user_by_id(debtor_id)
        if not debtor:
            print("User not found!")
            conn.close()
            return
        
        if creditor_id == debtor_id:
            print("A user cannot owe themselves money. Transaction cancelled.")
            conn.close()
            return
        
        # Amount owed
        amount = float(input("Enter the amount owed: "))
        if amount <= 0:
            print("Amount must be greater than zero.")
            conn.close()
            return
        
        # Transaction note
        note = input("Enter a note for this transaction (e.g., 'Loan', 'Debt repayment'): ")
        
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add transaction record
        cursor.execute("""
        INSERT INTO transactions (giver_name, giver_id, receiver_name, receiver_id, amount, date, note)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (debtor[2], debtor[1], creditor[2], creditor[1], amount, current_date, note))
        
        # Update balance (using sorted user IDs to ensure consistency)
        user_ids = sorted([creditor[1], debtor[1]])
        user_names = [get_user_by_id(user_ids[0])[2], get_user_by_id(user_ids[1])[2]]
        
        # Check if balance record exists
        cursor.execute("""
        SELECT * FROM balances WHERE (user1_id = ? AND user2_id = ?)
        """, (user_ids[0], user_ids[1]))
        
        balance = cursor.fetchone()
        
        if balance:
            # Update existing balance
            net_amount = balance[5]
            if creditor[1] == user_ids[0]:
                net_amount += amount
            else:
                net_amount -= amount
            
            cursor.execute("""
            UPDATE balances SET net_amount = ?, last_updated = ?
            WHERE user1_id = ? AND user2_id = ?
            """, (net_amount, current_date, user_ids[0], user_ids[1]))
        else:
            # Create new balance record
            net_amount = amount if creditor[1] == user_ids[0] else -amount
            
            cursor.execute("""
            INSERT INTO balances (user1_id, user1_name, user2_id, user2_name, net_amount, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (user_ids[0], user_names[0], user_ids[1], user_names[1], net_amount, current_date))
        
        conn.commit()
        print(f"\nDirect payment of ₹{amount:.2f} recorded successfully!")
        print(f"{debtor[2]} owes {creditor[2]} ₹{amount:.2f}")
        print(f"Note: {note}")
    
    conn.close()

# Balance checking functions
def check_balances():
    print("\n===== CHECK BALANCES =====")
    print("1. Check balances for a specific user")
    print("2. Check all balances")
    print("3. Check who owes me")
    print("4. Check who I owe")
    print("5. Back to main menu")
    
    choice = input("Enter your choice (1-5): ")
    
    if choice == '1':
        check_user_balances()
    elif choice == '2':
        check_all_balances()
    elif choice == '3':
        check_who_owes_me()
    elif choice == '4':
        check_who_i_owe()

def check_user_balances():
    print_all_users()
    user_id = input("\nEnter user ID to check balances: ")
    user = get_user_by_id(user_id)
    
    if not user:
        print("User not found!")
        return
    
    conn = sqlite3.connect('hisaab_db.sqlite')
    cursor = conn.cursor()
    
    # Get balances where user is involved
    cursor.execute("""
    SELECT b.*, GROUP_CONCAT(t.note, ' | ') as notes
    FROM balances b
    LEFT JOIN transactions t ON 
        ((t.giver_id = b.user1_id AND t.receiver_id = b.user2_id) OR 
         (t.giver_id = b.user2_id AND t.receiver_id = b.user1_id))
    WHERE b.user1_id = ? OR b.user2_id = ?
    GROUP BY b.sno
    """, (user_id, user_id))
    
    balances = cursor.fetchall()
    conn.close()
    
    if not balances:
        print(f"\n{user[2]} has no outstanding balances.")
        return
    
    print(f"\n===== BALANCES FOR {user[2].upper()} =====")
    for balance in balances:
        user1_id, user1_name = balance[1], balance[2]
        user2_id, user2_name = balance[3], balance[4]
        net_amount = balance[5]
        notes = balance[7] if len(balance) > 7 and balance[7] else "No notes"
        
        if user_id == user1_id:
            if net_amount > 0:
                print(f"{user2_name} owes {user[2]} ₹{abs(net_amount):.2f}")
                print(f"Notes: {notes}")
            elif net_amount < 0:
                print(f"{user[2]} owes {user2_name} ₹{abs(net_amount):.2f}")
                print(f"Notes: {notes}")
            else:
                print(f"{user[2]} and {user2_name} are settled up")
        else:  # user is user2
            if net_amount > 0:
                print(f"{user[2]} owes {user1_name} ₹{abs(net_amount):.2f}")
                print(f"Notes: {notes}")
            elif net_amount < 0:
                print(f"{user1_name} owes {user[2]} ₹{abs(net_amount):.2f}")
                print(f"Notes: {notes}")
            else:
                print(f"{user[2]} and {user1_name} are settled up")
        print("-" * 40)

def check_all_balances():
    conn = sqlite3.connect('hisaab_db.sqlite')
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT b.*, GROUP_CONCAT(t.note, ' | ') as notes
    FROM balances b
    LEFT JOIN transactions t ON 
        ((t.giver_id = b.user1_id AND t.receiver_id = b.user2_id) OR 
         (t.giver_id = b.user2_id AND t.receiver_id = b.user1_id))
    WHERE b.net_amount != 0
    GROUP BY b.sno
    """)
    
    balances = cursor.fetchall()
    
    conn.close()
    
    if not balances:
        print("\nNo outstanding balances in the system.")
        return
    
    print("\n===== ALL OUTSTANDING BALANCES =====")
    for balance in balances:
        user1_name, user2_name = balance[2], balance[4]
        net_amount = balance[5]
        notes = balance[7] if len(balance) > 7 and balance[7] else "No notes"
        
        if net_amount > 0:
            print(f"{user2_name} owes {user1_name} ₹{abs(net_amount):.2f}")
            print(f"Notes: {notes}")
        elif net_amount < 0:
            print(f"{user1_name} owes {user2_name} ₹{abs(net_amount):.2f}")
            print(f"Notes: {notes}")
        print("-" * 40)

def check_who_owes_me():
    print_all_users()
    user_id = input("\nEnter your user ID: ")
    user = get_user_by_id(user_id)
    
    if not user:
        print("User not found!")
        return
    
    conn = sqlite3.connect('hisaab_db.sqlite')
    cursor = conn.cursor()
    
    # Find balances where others owe this user
    cursor.execute("""
    SELECT b.*, GROUP_CONCAT(t.note, ' | ') as notes
    FROM balances b
    LEFT JOIN transactions t ON 
        ((t.giver_id = b.user1_id AND t.receiver_id = b.user2_id) OR 
         (t.giver_id = b.user2_id AND t.receiver_id = b.user1_id))
    WHERE ((b.user1_id = ? AND b.net_amount > 0) OR (b.user2_id = ? AND b.net_amount < 0))
    GROUP BY b.sno
    """, (user_id, user_id))
    
    balances = cursor.fetchall()
    conn.close()
    
    if not balances:
        print(f"\nNo one owes {user[2]} any money.")
        return
    
    print(f"\n===== PEOPLE WHO OWE {user[2].upper()} =====")
    for balance in balances:
        user1_id, user1_name = balance[1], balance[2]
        user2_id, user2_name = balance[3], balance[4]
        net_amount = balance[5]
        notes = balance[7] if len(balance) > 7 and balance[7] else "No notes"
        
        if user_id == user1_id:
            print(f"{user2_name} owes {user[2]} ₹{abs(net_amount):.2f}")
            print(f"Notes: {notes}")
        else:  # user is user2
            print(f"{user1_name} owes {user[2]} ₹{abs(net_amount):.2f}")
            print(f"Notes: {notes}")
        print("-" * 40)

def check_who_i_owe():
    print_all_users()
    user_id = input("\nEnter your user ID: ")
    user = get_user_by_id(user_id)
    
    if not user:
        print("User not found!")
        return
    
    conn = sqlite3.connect('hisaab_db.sqlite')
    cursor = conn.cursor()
    
    # Find balances where this user owes others
    cursor.execute("""
    SELECT b.*, GROUP_CONCAT(t.note, ' | ') as notes
    FROM balances b
    LEFT JOIN transactions t ON 
        ((t.giver_id = b.user1_id AND t.receiver_id = b.user2_id) OR 
         (t.giver_id = b.user2_id AND t.receiver_id = b.user1_id))
    WHERE ((b.user1_id = ? AND b.net_amount < 0) OR (b.user2_id = ? AND b.net_amount > 0))
    GROUP BY b.sno
    """, (user_id, user_id))
    
    balances = cursor.fetchall()
    conn.close()
    
    if not balances:
        print(f"\n{user[2]} doesn't owe anyone any money.")
        return
    
    print(f"\n===== PEOPLE {user[2].upper()} OWES =====")
    for balance in balances:
        user1_id, user1_name = balance[1], balance[2]
        user2_id, user2_name = balance[3], balance[4]
        net_amount = balance[5]
        notes = balance[7] if len(balance) > 7 and balance[7] else "No notes"
        
        if user_id == user1_id:
            print(f"{user[2]} owes {user2_name} ₹{abs(net_amount):.2f}")
            print(f"Notes: {notes}")
        else:  # user is user2
            print(f"{user[2]} owes {user1_name} ₹{abs(net_amount):.2f}")
            print(f"Notes: {notes}")
        print("-" * 40)

# Main application
def main():
    # Setup database on first run
    setup_database()
    
    while True:
        print("\n===== HISAAB APP - EXPENSE SPLITTING SYSTEM =====")
        print("1. Check balances")
        print("2. Add new expense")
        print("3. Add new user")
        print("4. View all users")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            check_balances()
        elif choice == '2':
            add_transaction()
        elif choice == '3':
            add_user()
        elif choice == '4':
            print_all_users()
        elif choice == '5':
            print("Thank you for using Hisaab App. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()