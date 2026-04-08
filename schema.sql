CREATE DATABASE IF NOT EXISTS `caperclub`
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE `caperclub`;

CREATE TABLE IF NOT EXISTS time_slots (
    id CHAR(36) NOT NULL,
    name VARCHAR(120) NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_time_slots_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS users (
    id CHAR(36) NOT NULL,
    name VARCHAR(120) NOT NULL,
    email VARCHAR(190) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('ADMIN', 'USER') NOT NULL,
    mobile_number VARCHAR(15) NULL,
    member_id VARCHAR(64) NOT NULL,
    slot_id CHAR(36) NULL,
    sport VARCHAR(64) NOT NULL DEFAULT 'General',
    membership_plan VARCHAR(32) NOT NULL DEFAULT 'Monthly',
    membership_level VARCHAR(120) NOT NULL DEFAULT '',
    membership_start DATE NULL,
    membership_expiry DATE NULL,
    payment_amount DECIMAL(10, 2) NOT NULL DEFAULT 0,
    due_amount DECIMAL(10, 2) NOT NULL DEFAULT 0,
    payment_mode VARCHAR(16) NOT NULL DEFAULT 'UPI',
    payment_status VARCHAR(16) NOT NULL DEFAULT 'Pending',
    last_action VARCHAR(8) NULL,
    last_action_at DATETIME NULL,
    note TEXT NOT NULL,
    face_images_count INT NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_users_email (email),
    UNIQUE KEY uq_users_member_id (member_id),
    KEY idx_users_mobile_number (mobile_number),
    KEY idx_users_slot_id (slot_id),
    CONSTRAINT fk_users_slot_id
      FOREIGN KEY (slot_id) REFERENCES time_slots(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS face_embeddings (
    id INT NOT NULL AUTO_INCREMENT,
    user_id CHAR(36) NOT NULL,
    image_data LONGTEXT NOT NULL,
    embedding_vector LONGBLOB NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_face_embeddings_user_id (user_id),
    CONSTRAINT fk_face_embeddings_user_id
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS sessions (
    id CHAR(36) NOT NULL,
    user_id CHAR(36) NOT NULL,
    slot_id CHAR(36) NULL,
    area VARCHAR(120) NOT NULL,
    status ENUM('ACTIVE', 'ENDED', 'EXPIRED', 'DENIED') NOT NULL DEFAULT 'ACTIVE',
    confidence DOUBLE NOT NULL DEFAULT 0,
    started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at DATETIME NULL,
    slot_start_at DATETIME NULL,
    slot_end_at DATETIME NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_sessions_user_id (user_id),
    KEY idx_sessions_slot_id (slot_id),
    CONSTRAINT fk_sessions_user_id
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_sessions_slot_id
      FOREIGN KEY (slot_id) REFERENCES time_slots(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS user_timelines (
    id CHAR(36) NOT NULL,
    user_id CHAR(36) NOT NULL,
    event_type ENUM('ENTRY', 'EXIT', 'DENIED') NOT NULL,
    area VARCHAR(120) NOT NULL,
    occurred_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    total_minutes INT NULL,
    note TEXT NOT NULL,
    PRIMARY KEY (id),
    KEY idx_user_timelines_user_id (user_id),
    CONSTRAINT fk_user_timelines_user_id
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS payment_history (
    id CHAR(36) NOT NULL,
    user_id CHAR(36) NOT NULL,
    plan VARCHAR(32) NOT NULL,
    amount DECIMAL(10, 2) NOT NULL DEFAULT 0,
    payment_mode VARCHAR(16) NOT NULL DEFAULT 'UPI',
    payment_status VARCHAR(16) NOT NULL DEFAULT 'Pending',
    membership_start DATE NULL,
    membership_expiry DATE NULL,
    source VARCHAR(120) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_payment_history_user_id (user_id),
    CONSTRAINT fk_payment_history_user_id
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS announcements (
    id CHAR(36) NOT NULL,
    title VARCHAR(120) NOT NULL,
    message TEXT NOT NULL,
    tone ENUM('BLUE', 'PURPLE', 'GREEN', 'RED', 'AMBER') NOT NULL DEFAULT 'BLUE',
    user_id CHAR(36) NULL,
    created_by_id CHAR(36) NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_announcements_user_id (user_id),
    KEY idx_announcements_created_by_id (created_by_id),
    CONSTRAINT fk_announcements_user_id
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    CONSTRAINT fk_announcements_created_by_id
      FOREIGN KEY (created_by_id) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS notifications (
    id CHAR(36) NOT NULL,
    user_id CHAR(36) NOT NULL,
    title VARCHAR(120) NOT NULL,
    message TEXT NOT NULL,
    tone ENUM('BLUE', 'PURPLE', 'GREEN', 'RED', 'AMBER') NOT NULL DEFAULT 'BLUE',
    is_read TINYINT(1) NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_notifications_user_id (user_id),
    CONSTRAINT fk_notifications_user_id
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
