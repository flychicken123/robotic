# Package Sorting System

A Java implementation of a robotic package sorting system that categorizes packages based on their dimensions and mass.

## Overview

This system is designed for Thoughtful's robotic automation factory to automatically sort packages into different stacks based on their physical characteristics. The sorting is performed using specific criteria for package dimensions and mass.

## Sorting Criteria

### Package Classification

1. **Bulky Package** (if either condition is met):
   - Volume ≥ 1,000,000 cm³ (width × height × length)
   - Any dimension (width, height, or length) ≥ 150 cm

2. **Heavy Package**:
   - Mass ≥ 20 kg

### Stack Categories

- **STANDARD**: Regular packages that are neither bulky nor heavy
- **SPECIAL**: Packages that are either bulky OR heavy
- **REJECTED**: Packages that are BOTH bulky AND heavy

## Implementation

The system consists of two main methods:

### 1. `isBulky(int width, int height, int length)`
- Checks if a package is considered bulky
- Returns `true` if:
  - Any dimension is ≥ 150 cm
  - Volume is ≥ 1,000,000 cm³

### 2. `isHeavy(int mass)`
- Checks if a package is considered heavy
- Returns `true` if mass is ≥ 20 kg

### Main Sorting Method
```java
public static String sort(int width, int height, int length, int mass)
```

## Usage Example

```java
// Example 1: Standard package
System.out.println(sort(90, 100, 100, 10));  // Output: "STANDARD"

// Example 2: Special package (bulky)
System.out.println(sort(150, 100, 100, 10));  // Output: "SPECIAL"

// Example 3: Special package (heavy)
System.out.println(sort(100, 100, 100, 20));  // Output: "SPECIAL"

// Example 4: Rejected package (both bulky and heavy)
System.out.println(sort(150, 100, 100, 20));  // Output: "REJECTED"
```

## Technical Details

- Dimensions are measured in centimeters (cm)
- Mass is measured in kilograms (kg)
- Volume is calculated in cubic centimeters (cm³)
- The system uses integer values for all measurements

## Constants

```java
private static final int MAX_VOLUME = 1_000_000;    // Maximum volume threshold
private static final int MAX_DIMENSION = 150;       // Maximum dimension threshold
private static final int MAX_MASS = 20;             // Maximum mass threshold
```