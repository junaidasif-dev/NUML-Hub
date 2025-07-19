-- Create the database
CREATE DATABASE OPD;

-- Use the newly created database
USE OPD;

-- Create the Patient table
CREATE TABLE Patient (
    PatID INT PRIMARY KEY,
    PatName VARCHAR(50),
    PatAge INT,
    PatContact VARCHAR(15)
);

-- Create the Doctor table
CREATE TABLE Doctor (
    DrID INT PRIMARY KEY,
    DRname VARCHAR(50),
    DrContact VARCHAR(15),
    DrFee DECIMAL(8, 2)
);

-- Create the Appointment table
CREATE TABLE Appointment (
    AptID INT PRIMARY KEY,
    PatID INT,
    DrID INT,
    AptDate DATE,
    Status ENUM('Scheduled', 'Cancelled', 'Completed'),
    AptTime TIME,
    FOREIGN KEY (PatID) REFERENCES Patient(PatID),
    FOREIGN KEY (DrID) REFERENCES Doctor(DrID)
);

-- Create the Payment table
CREATE TABLE Payment (
    PayID INT PRIMARY KEY,
    PatID INT,
    Pay_Date_Time DATETIME,
    PayAmount DECIMAL(8, 2),
    PayMethod VARCHAR(25),
    FOREIGN KEY (PatID) REFERENCES Patient(PatID)
);

-- Create the Remuneration table
CREATE TABLE Remuneration (
    RemID INT PRIMARY KEY,
    DrID INT,
    Contact_Hrs INT,
    Month VARCHAR(15),
    RemAmount DECIMAL(10, 2),
    FOREIGN KEY (DrID) REFERENCES Doctor(DrID)
);

-- Create the Slot table
CREATE TABLE Slot (
    SlotID INT PRIMARY KEY,
    DrID INT,
    SlotDate DATE,
    SlotTime_start TIME,
    SlotTime_end TIME,
    Status ENUM('Unavailable', 'Available', 'Booked'),
    FOREIGN KEY (DrID) REFERENCES Doctor(DrID)
);

INSERT INTO Patient(PatID, PatName, PatAge, PatContact) VALUES
(1, "Qasim", 20, "0300-1213222"), 
(2, "Ali", 21, "0310-2132156"),
(3, "Tahir", 17, "0321-1364998"),
(4, "Adeel", 19, "0342-2332002"),
(5, "Haris", 22, "0314-9434776");

INSERT INTO Doctor(DrID, DrName, DrContact, DrFee) VALUES
(100, "Hamdan", "0309-1213982", 1500.00), 
(200, "Khizar", "0313-2131456", 1000.00),
(300, "Shami", "0331-1376998", 1500.00),
(400, "Usman", "0345-2002002", 2000.00),
(500, "Umair", "0317-9409776", 500.00);

INSERT INTO Appointment(AptID, PatID, DrID, AptDate, Status, AptTime) VALUES
(1000, 1, 200, "2024-5-1", "Scheduled", "08:15:00"),
(2000, 2, 300, "2024-2-1", "Completed", "10:00:00"),
(3000, 3, 100, "2024-5-4", "Scheduled", "13:30:00"),
(4000, 2, 500, "2024-5-7", "Scheduled", "11:45:00"),
(5000, 4, 400, "2024-6-10", "Scheduled", "15:00:00"),
(6000, 5, 100, "2024-5-7", "Cancelled", "9:00:00");

INSERT INTO Payment(PayID, PatID, Pay_Date_Time, PayAmount, PayMethod) VALUES
(10000, 2, "2024-02-01 10:30:00", 1500.00, "Credit Card"),
(20000, 3, "2024-05-04 13:45:00", 1500.00, "Cash"),
(30000, 4, "2024-06-10 15:15:00", 2000.00, "Debit Card"),
(40000, 1, "2024-05-01 08:30:00", 1000.00, "Credit Card"),
(50000, 2, "2024-05-07 12:00:00", 500.00, "Cash");

INSERT INTO Slot(SlotID, DrID, SlotDate, SlotTime_start, SlotTime_end, Status) VALUES
(100000, 100, "2024-05-04", "13:30:00", "13:45:00", "Booked"),       
(200000, 200, "2024-05-01", "11:00:00", "11:15:00", "Unavailable"),  
(300000, 300, "2024-05-01", "10:15:00", "10:30:00", "Available"),    
(400000, 400, "2024-06-10", "15:00:00", "15:15:00", "Booked"),       
(500000, 500, "2024-05-07", "09:15:00", "09:30:00", "Available");    

INSERT INTO Remuneration(RemID, DrID, Contact_Hrs, Month, RemAmount) VALUES
(1, 100, 10, 'May 2024', 60000.00),  
(2, 200, 12, 'Feb 2024', 48000.00),  
(3, 300, 8, 'May 2024', 48000.00),   
(4, 400, 15, 'June 2024', 120000.00),
(5, 500, 20, 'May 2024', 40000.00);  