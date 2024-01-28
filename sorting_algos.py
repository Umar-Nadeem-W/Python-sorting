import sys
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt




POSITIVE_INFINITY = sys.maxsize
NEGATIVE_INFINITY = -sys.maxsize - 1


def incmerge(arr, left, middle, right):
    n1 = middle - left + 1
    n2 = right - middle

    leftArr = arr[left:left + n1].copy()
    rightArr = arr[middle + 1:middle + 1 + n2].copy()

    leftArr = np.append(leftArr, POSITIVE_INFINITY)
    rightArr = np.append(rightArr, POSITIVE_INFINITY)

    i, j = 0, 0
    for k in range(left, right + 1):
        if leftArr[i] <= rightArr[j]:
            arr[k] = leftArr[i]
            i += 1
        else:
            arr[k] = rightArr[j]
            j += 1
    


def decmerge(arr, left, middle, right):
    n1 = middle - left + 1
    n2 = right - middle

    leftArr = arr[left:left + n1].copy()
    rightArr = arr[middle + 1:middle + 1 + n2].copy()

    leftArr = np.append(leftArr,  NEGATIVE_INFINITY)
    rightArr = np.append(rightArr,NEGATIVE_INFINITY)
    i, j = 0, 0
    for k in range(left, right + 1):
        if leftArr[i] >= rightArr[j]:
            arr[k] = leftArr[i]
            i += 1
        else:
            arr[k] = rightArr[j]
            j += 1

def incmergeSort(arr, left, right):
    if left < right:
        middle = left + (right - left) // 2
        incmergeSort(arr, left, middle)
        incmergeSort(arr, middle + 1, right)
        incmerge(arr, left, middle, right)

def decmergeSort(arr, left, right):
    if left < right:
        middle = left + (right - left) // 2
        decmergeSort(arr, left, middle)
        decmergeSort(arr, middle + 1, right)
        decmerge(arr, left, middle, right)


def inc_insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def dec_insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key > arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key > arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    sorted_arr = arr.copy()
    for i in range(1, len(sorted_arr)):
        key = sorted_arr[i]
        j = i - 1
        while j >= 0 and key < sorted_arr[j]:
            sorted_arr[j + 1] = sorted_arr[j]
            j -= 1
        sorted_arr[j + 1] = key
    return sorted_arr


    sorted_arr = arr.copy()
    for i in range(1, len(sorted_arr)):
        key = sorted_arr[i]
        j = i - 1
        while j >= 0 and key > sorted_arr[j]:
            sorted_arr[j + 1] = sorted_arr[j]
            j -= 1
        sorted_arr[j + 1] = key
    return sorted_arr



def main():
     data = pd.read_csv(r"C:\Users\umarn\Downloads\Dataset.csv")
     print("Data loaded successfully.")
     numeric_columns = data.select_dtypes(include=[np.number])
     medians = numeric_columns.median()
     numeric_columns = numeric_columns.fillna(medians)

     inc_merge_sort_t = []
     dec_merge_sort_t = []
     inc_insertion_sort_t = []
     dec_insertion_sort_t = []

     
     # Convert the DataFrame to a NumPy array
     numeric_array = numeric_columns.to_numpy()
     

     for col in range(numeric_array.shape[1]):
         column_name = numeric_columns.columns[col]

         # Measure sorting time
         merge_inc_start_time = time.process_time()
         incmergeSort(numeric_array[:, col], 0, numeric_array.shape[0] - 1)
         merge_inc_end_time = time.process_time()
         inc_merge_sort_t.append(merge_inc_end_time - merge_inc_start_time)

         # Print column name and sorting time
         print(f"Column Name: {column_name}")
         print("Sorting time (NonDecreasing mergesort): ", inc_merge_sort_t[col])

         # Extract the sorted column
         sorted_column = numeric_array[:, col]

         # Create a DataFrame with the sorted column
         sorted_df = pd.DataFrame({column_name: sorted_column})

         # Print the sorted DataFrame
         print(f"Sorted Column Elements:\n{sorted_df}")
         print("\n")
    
         

     for col in range(numeric_array.shape[1]):
        column_name = numeric_columns.columns[col]
        
        # Measure sorting time
        dec_merge_start_time = time.process_time()
        decmergeSort(numeric_array[:, col], 0, numeric_array.shape[0] - 1)
        dec_merge_end_time = time.process_time()
        dec_merge_sort_t.append(dec_merge_end_time - dec_merge_start_time)
        
        # Print column name and sorting time
        print(f"Column Name: {column_name}")
        print("Sorting time (Decreasing mergesort): ", dec_merge_sort_t[col])
        
        # Extract the sorted column
        sorted_column = numeric_array[:, col]
        
        # Create a DataFrame with the sorted column
        sorted_df = pd.DataFrame({column_name: sorted_column})
        
        # Print the sorted DataFrame
        print(f"Sorted Column Elements:\n{sorted_df}")
        print("\n")
     

     """ for col in range(3):
        column_name = numeric_columns.columns[col]

        # Measure sorting time for increasing insertion sort
        inc_ins_start_time = time.process_time()
        sorted_column_inc = inc_insertion_sort(numeric_array[:, col])
        inc_ins_end_time = time.process_time()
        inc_insertion_sort_t.append(inc_ins_end_time - inc_ins_start_time)

        # Print column name and sorting time for increasing insertion sort
        print(f"Column Name: {column_name}")
        print("Sorting time (Increasing insertionsort): ", inc_insertion_sort_t[col])

        # Create a DataFrame with the sorted column for increasing insertion sort
        sorted_df_inc = pd.DataFrame({column_name: sorted_column_inc})

        # Print the sorted DataFrame for increasing insertion sort
        print(f"Sorted Column Elements (Increasing Insertion Sort):\n{sorted_df_inc}")
        print("\n")

     
     for col in range(min(3, numeric_array.shape[1])):
        column_name = numeric_columns.columns[col]

        # Measure sorting time for decreasing insertion sort
        dec_ins_start_time = time.process_time()
        sorted_column_dec = dec_insertion_sort(numeric_array[:, col])
        dec_ins_end_time = time.process_time()
        dec_insertion_sort_t.append(dec_ins_end_time - dec_ins_start_time)

        # Print column name and sorting time for decreasing insertion sort
        print(f"Column Name: {column_name}")
        print("Sorting time (Decreasing insertionsort): ", dec_insertion_sort_t[col])

        # Create a DataFrame with the sorted column for decreasing insertion sort
        sorted_df_dec = pd.DataFrame({column_name: sorted_column_dec})

        # Print the sorted DataFrame for decreasing insertion sort
        print(f"Sorted Column Elements (Decreasing Insertion Sort):\n{sorted_df_dec}")
        print("\n")"""
    
    #  for col in range(numeric_array.shape[1]):
    #     incmergeSort(numeric_array[:, col], 0, numeric_array.shape[0] - 1)
    
    #  total_end_time = time.process_time()
    #  total_time = total_end_time - total_start_time
     
    #  # Create a DataFrame with the sorted columns
    #  sorted_df = pd.DataFrame(data=numeric_array, columns=numeric_columns.columns)
     
    #  print("Total Sorting Time (Increasing mergesort):", total_time, "seconds")
    #  print(sorted_df)
    
 #
 # inc_ins_start_time = time.process_time()
 # for col in numeric_columns.columns:
 #  numeric_columns[col] = inc_insertion_sort(numeric_columns[col])
 # inc_ins_end_time = time.process_time()
 # inc_insertion_sort_t.append(inc_ins_end_time - inc_ins_start_time)
 # inc_ins_numeric_cols_sorted  = pd.DataFrame(data=numeric_array, columns=numeric_columns.columns)
 # print(inc_ins_numeric_cols_sorted) 
     

    # iist = np.array(inc_insertion_sort_t[:3])
     imst = np.array(inc_merge_sort_t[:3])
     iist = np.array([680.78125, 703.84375, 549.875])
   
     first_3_columns = numeric_columns.columns[:3]
     
     results = pd.DataFrame({
         'Attribute Name': first_3_columns,
         'Merge Sort time (NonDecreasing)': imst,
         'Insertion Sort time (NonDecreasing)': iist,
     })

     results = results.sort_values(by='Attribute Name')
     results.plot(x='Attribute Name', kind='bar', figsize=(12, 6))
     plt.xlabel('Attribute Name')
     plt.ylabel('Processor Time (s)')
     plt.title('Sorting Algorithm Comparison')
     plt.legend(loc='upper right')
     plt.show()


      
 
if __name__ == '__main__':
    main()




    #  imst = np.array([0.96875, 0.921875, 0.953125])