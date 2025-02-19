import tkinter as tk
from tkinter import messagebox, simpledialog
import json
import datetime

TASKS_FILE = "tasks.json"

def load_tasks():
    try:
        with open(TASKS_FILE, "r") as file:
            data = json.load(file)
            if data["date"] != str(datetime.date.today()):
                return {"date": str(datetime.date.today()), "tasks": []}
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {"date": str(datetime.date.today()), "tasks": []}

def save_tasks(tasks):
    with open(TASKS_FILE, "w") as file:
        json.dump(tasks, file, indent=4)

def modify_task(task_index, action):
    tasks = load_tasks()
    if 0 <= task_index < len(tasks["tasks"]):
        if tasks["tasks"][task_index]["completed"]:
            now = datetime.datetime.now()
            remaining_time = (datetime.datetime.combine(now.date(), datetime.time(23, 59, 59)) - now).seconds
            if remaining_time > 0:
                messagebox.showerror("Error", f"Cannot {action} a completed task before EOD.")
                return
        if action == "delete":
            del tasks["tasks"][task_index]
        elif action == "update":
            new_task = simpledialog.askstring("Input", "Enter new task description:")
            if new_task:
                tasks["tasks"][task_index]["task"] = new_task
        save_tasks(tasks)
        refresh_task_list()
    else:
        messagebox.showerror("Error", "Invalid task index!")

def add_task(task):
    tasks = load_tasks()
    tasks["tasks"].append({"task": task, "completed": False})
    save_tasks(tasks)
    refresh_task_list()

def refresh_task_list():
    for widget in task_frame.winfo_children():
        widget.destroy()
    tasks = load_tasks()
    for idx, task in enumerate(tasks["tasks"]):
        var = tk.BooleanVar(value=task["completed"])
        chk = tk.Checkbutton(task_frame, text=f"{idx + 1}. {task['task']}", variable=var, command=lambda idx=idx, var=var: update_task_status(idx, var))
        chk.pack(anchor='w')
        if task["completed"]:
            chk.config(state=tk.DISABLED)

def update_task_status(task_index, var):
    tasks = load_tasks()
    tasks["tasks"][task_index]["completed"] = var.get()
    save_tasks(tasks)
    refresh_task_list()

def start_timer():
    def update_timer():
        now = datetime.datetime.now()
        remaining_time = (datetime.datetime.combine(now.date(), datetime.time(23, 59, 59)) - now).seconds
        timer_label.config(text=f"Time remaining today: {remaining_time // 3600}h {(remaining_time % 3600) // 60}m {remaining_time % 60}s")
        timer_label.after(1000, update_timer)
    update_timer()

def main():
    global task_frame, timer_label

    root = tk.Tk()
    root.title("To-Do List")

    frame = tk.Frame(root)
    frame.pack(pady=10)

    tk.Button(frame, text="Add Task", command=lambda: add_task(simpledialog.askstring("Input", "Enter task description:"))).pack(side=tk.LEFT, padx=5)
    tk.Button(frame, text="Delete Task", command=lambda: modify_task(simpledialog.askinteger("Input", "Enter task number to delete:") - 1, "delete")).pack(side=tk.LEFT, padx=5)
    tk.Button(frame, text="Update Task", command=lambda: modify_task(simpledialog.askinteger("Input", "Enter task number to update:") - 1, "update")).pack(side=tk.LEFT, padx=5)
    tk.Button(frame, text="Exit", command=root.quit).pack(side=tk.LEFT, padx=5)

    task_frame = tk.Frame(root)
    task_frame.pack(pady=10)

    timer_label = tk.Label(root, text="")
    timer_label.pack(pady=10)

    refresh_task_list()
    start_timer()

    root.mainloop()

if __name__ == "__main__":
    main()