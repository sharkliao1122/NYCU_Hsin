import random

number = random.randint(1, 10)
low = 1
high = 10
max_attempts = 3
guessed = set()

for i in range(max_attempts):
    while True:
        try:
            num_guess = int(input(f"請猜一個{low}到{high}的數字 (剩餘{max_attempts-i}次): "))
        except ValueError:
            print("請輸入有效的整數！")
            continue
        if num_guess < low or num_guess > high:
            print("請輸入有效的數字！")
            print(f"請輸入{low}到{high}之間的數字！")
            continue
        if num_guess in guessed:
            print("你已經猜過這個數字了，請換一個！")
            continue
        break
    guessed.add(num_guess)

    if num_guess == number:
        print("恭喜你猜對了!")
        break
    else:
        if num_guess < number:
            low = max(low, num_guess + 1)
            print(f"很遺憾你猜錯了! 提示: 答案比{num_guess}大")
        else:
            high = min(high, num_guess - 1)
            print(f"很遺憾你猜錯了! 提示: 答案比{num_guess}小")
        if low > high:
            print("範圍已經不合理，遊戲結束！")
            break
        if i == max_attempts - 1:
            print(f"很遺憾你猜錯了! 正確答案是 {number}")
            break
    