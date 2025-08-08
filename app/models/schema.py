def get_schema():
    return {
        'users': """
            CREATE TABLE IF NOT EXISTS users (
            id CHAR(36) PRIMARY KEY,
            name VARCHAR(100),
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash TEXT,
            is_google_user BOOLEAN DEFAULT FALSE,
            remaining_queries INT DEFAULT 0,
            status BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
        """,
        'chat_sessions': """
            CREATE TABLE IF NOT EXISTS chat_sessions (
            id CHAR(36) PRIMARY KEY,
            user_id CHAR(36) NOT NULL,
            session_name VARCHAR(255),
            status BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        ) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
        """,
        'messages': """
            CREATE TABLE IF NOT EXISTS messages (
            id CHAR(36) PRIMARY KEY,
            session_id CHAR(36) NOT NULL,
            sender ENUM('user', 'bot') NOT NULL,
            message TEXT NOT NULL,
            stream_chunk BOOLEAN DEFAULT FALSE,
            status BOOLEAN DEFAULT TRUE,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
        ) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
        """,
        'subscriptions': """
            CREATE TABLE IF NOT EXISTS subscriptions (
            id CHAR(36) PRIMARY KEY,
            package_name VARCHAR(100) NOT NULL,
            start_date TIMESTAMP NULL DEFAULT NULL,
            end_date TIMESTAMP NULL DEFAULT NULL,
            max_queries_per_day INT NOT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            status BOOLEAN DEFAULT TRUE
        ) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
        """,
        'transactions': """
            CREATE TABLE IF NOT EXISTS transactions (
            id CHAR(36) PRIMARY KEY,
            user_id CHAR(36) NOT NULL,
            subscription_id CHAR(36) NOT NULL,
            payment_method ENUM('jazzcash', 'easypaisa', 'credit_card') NOT NULL,
            amount INT NOT NULL,
            status BOOLEAN DEFAULT TRUE,
            payment_status ENUM('pending', 'success', 'failed') DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (subscription_id) REFERENCES subscriptions(id)
        ) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
        """
    }