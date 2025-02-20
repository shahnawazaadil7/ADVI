from matrix import DynamicMatrix

def main():
    matrix = DynamicMatrix.from_user_input()
    
    while True:
        choice = input("\n1. Display\n2. Insert Row\n3. Insert Column\n4. Delete Row\n5. Delete Column\n6. Update Element\n7. Multiply\n8. Transpose\n9. Exit\nChoice: ")
        
        if choice == "1":
            matrix.display()
        elif choice == "2":
            matrix.handle_insert_row()
        elif choice == "3":
            matrix.handle_insert_column()
        elif choice == "4":
            matrix.handle_delete_row()
        elif choice == "5":
            matrix.handle_delete_column()
        elif choice == "6":
            matrix.handle_update_element()
        elif choice == "7":
            matrix.handle_multiply()
        elif choice == "8":
            matrix.transpose().display()
        elif choice == "9":
            print("Exiting...")
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()
