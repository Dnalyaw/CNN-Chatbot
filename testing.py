import time

def loading_animation():
    print("Loading", end="")
    for _ in range(5):  # Print 5 sets of dots
        print(".", end="", flush=True)  # Print one dot at a time
        time.sleep(0.5)  # Wait for 0.5 seconds before printing the next dot
    print()  # Print a new line after the animation

# Call the function
while True:
    loading_animation()