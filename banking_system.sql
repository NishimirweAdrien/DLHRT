-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jan 08, 2026 at 08:15 AM
-- Server version: 10.4.27-MariaDB
-- PHP Version: 8.1.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `banking_system`
--

-- --------------------------------------------------------

--
-- Table structure for table `acc_table`
--

CREATE TABLE `acc_table` (
  `acc_id` int(11) NOT NULL,
  `user_id` int(11) DEFAULT NULL,
  `acc_no` varchar(255) DEFAULT NULL,
  `accType` varchar(11) NOT NULL,
  `status` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `acc_table`
--

INSERT INTO `acc_table` (`acc_id`, `user_id`, `acc_no`, `accType`, `status`) VALUES
(1, 9, 'Bk/9', '1', 1),
(2, 10, 'Bk/10', '1', 1);

-- --------------------------------------------------------

--
-- Table structure for table `clients`
--

CREATE TABLE `clients` (
  `cl_id` int(11) NOT NULL,
  `first_name` varchar(250) DEFAULT NULL,
  `last_name` varchar(250) DEFAULT NULL,
  `account_no` double DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `clients`
--

INSERT INTO `clients` (`cl_id`, `first_name`, `last_name`, `account_no`) VALUES
(1, 'Adrien', 'NISHIMIRWE', 40011132029);

-- --------------------------------------------------------

--
-- Table structure for table `customers`
--

CREATE TABLE `customers` (
  `cust_id` int(11) NOT NULL,
  `first_name` varchar(50) DEFAULT NULL,
  `surname` varchar(50) DEFAULT NULL,
  `account_no` varchar(250) DEFAULT NULL,
  `amount` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `customers`
--

INSERT INTO `customers` (`cust_id`, `first_name`, `surname`, `account_no`, `amount`) VALUES
(1, 'Adrien', 'NISHIMIRWE', '40011132029', 5000);

-- --------------------------------------------------------

--
-- Table structure for table `laon_payment`
--

CREATE TABLE `laon_payment` (
  `payt_id` int(11) NOT NULL,
  `customer_id` int(11) DEFAULT NULL,
  `laonId` int(11) DEFAULT NULL,
  `done_by` int(11) DEFAULT NULL,
  `amount` double DEFAULT NULL,
  `status` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `laon_payment`
--

INSERT INTO `laon_payment` (`payt_id`, `customer_id`, `laonId`, `done_by`, `amount`, `status`) VALUES
(1, 9, 1, 1, 3000, 1);

-- --------------------------------------------------------

--
-- Table structure for table `tbl_loans`
--

CREATE TABLE `tbl_loans` (
  `loan_id` int(11) NOT NULL,
  `customer_id` int(11) DEFAULT NULL,
  `income_source` varchar(255) DEFAULT NULL,
  `monthly_income` varchar(255) DEFAULT NULL,
  `loan_amount` double DEFAULT NULL,
  `loan_to_be_paid` double DEFAULT NULL,
  `interest_rate` int(11) DEFAULT NULL,
  `loan_term` int(11) DEFAULT NULL,
  `collateral` varchar(250) DEFAULT NULL,
  `done_by` int(11) DEFAULT NULL,
  `paid_period` date DEFAULT NULL,
  `done_at` date DEFAULT current_timestamp(),
  `status` int(11) DEFAULT NULL,
  `purpase_loan` varchar(250) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `tbl_loans`
--

INSERT INTO `tbl_loans` (`loan_id`, `customer_id`, `income_source`, `monthly_income`, `loan_amount`, `loan_to_be_paid`, `interest_rate`, `loan_term`, `collateral`, `done_by`, `paid_period`, `done_at`, `status`, `purpase_loan`) VALUES
(1, 9, 'salary', '400000', 200000, 206731, 5, 15, 'Car', 1, '2026-06-13', '2025-03-13', 1, 'Business expansion');

-- --------------------------------------------------------

--
-- Table structure for table `transactions`
--

CREATE TABLE `transactions` (
  `transaction_id` int(11) NOT NULL,
  `user_id` int(11) DEFAULT NULL,
  `transaction_type` varchar(11) DEFAULT NULL,
  `amount` double DEFAULT NULL,
  `created_by` int(11) DEFAULT NULL,
  `status` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `transactions`
--

INSERT INTO `transactions` (`transaction_id`, `user_id`, `transaction_type`, `amount`, `created_by`, `status`) VALUES
(1, 9, 'deposit', 1000, NULL, NULL),
(2, 9, 'deposit', 40000, 1, NULL),
(3, 9, 'withdrawal', 3000, 1, NULL),
(4, 10, 'deposit', 1000, NULL, NULL);

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `user_id` int(11) NOT NULL,
  `username` varchar(20) DEFAULT NULL,
  `password` varchar(50) DEFAULT NULL,
  `logged_in` datetime NOT NULL DEFAULT current_timestamp(),
  `expire` datetime NOT NULL DEFAULT current_timestamp(),
  `user_email` varchar(50) DEFAULT NULL,
  `first_name` varchar(50) DEFAULT NULL,
  `last_name` varchar(50) DEFAULT NULL,
  `phone_number` varchar(50) DEFAULT NULL,
  `nid` varchar(60) DEFAULT NULL,
  `gender` varchar(11) DEFAULT NULL,
  `dob` date DEFAULT NULL,
  `location` varchar(30) DEFAULT NULL,
  `role_id` int(11) DEFAULT NULL,
  `status` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `acc_table`
--
ALTER TABLE `acc_table`
  ADD PRIMARY KEY (`acc_id`);

--
-- Indexes for table `clients`
--
ALTER TABLE `clients`
  ADD PRIMARY KEY (`cl_id`);

--
-- Indexes for table `customers`
--
ALTER TABLE `customers`
  ADD PRIMARY KEY (`cust_id`);

--
-- Indexes for table `laon_payment`
--
ALTER TABLE `laon_payment`
  ADD PRIMARY KEY (`payt_id`);

--
-- Indexes for table `tbl_loans`
--
ALTER TABLE `tbl_loans`
  ADD PRIMARY KEY (`loan_id`);

--
-- Indexes for table `transactions`
--
ALTER TABLE `transactions`
  ADD PRIMARY KEY (`transaction_id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`user_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `acc_table`
--
ALTER TABLE `acc_table`
  MODIFY `acc_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT for table `clients`
--
ALTER TABLE `clients`
  MODIFY `cl_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT for table `customers`
--
ALTER TABLE `customers`
  MODIFY `cust_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- AUTO_INCREMENT for table `laon_payment`
--
ALTER TABLE `laon_payment`
  MODIFY `payt_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `tbl_loans`
--
ALTER TABLE `tbl_loans`
  MODIFY `loan_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `transactions`
--
ALTER TABLE `transactions`
  MODIFY `transaction_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=5;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `user_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=11;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
