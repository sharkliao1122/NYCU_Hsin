import random

def binary_search_player(secret_number):
    """
    Simulates a single game using the binary search strategy.
    Returns (True/False, guess_log) where guess_log is a list of guess/result strings.
    """
    low = 1
    high = 10
    guess_log = []

    # --- Attempt 1 ---
    guess1 = 5
    if guess1 == secret_number:
        guess_log.append(f"猜 {guess1}: 恭喜猜對!")
        return True, guess_log
    else:
        if guess1 < secret_number:
            guess_log.append(f"猜 {guess1}: 錯，答案比{guess1}大")
        else:
            guess_log.append(f"猜 {guess1}: 錯，答案比{guess1}小")

    # --- Attempt 2 ---
    if guess1 < secret_number: # Secret is in [6, 10]
        guess2 = 8
    else: # Secret is in [1, 4]
        guess2 = 3

    if guess2 == secret_number:
        guess_log.append(f"猜 {guess2}: 恭喜猜對!")
        return True, guess_log
    else:
        if guess2 < secret_number:
            guess_log.append(f"猜 {guess2}: 錯，答案比{guess2}大")
        else:
            guess_log.append(f"猜 {guess2}: 錯，答案比{guess2}小")

    # --- Attempt 3 ---
    # Path 1: Secret was > 5
    if guess1 < secret_number:
        if guess2 < secret_number: # Secret was > 8, now in [9, 10]
            guess3 = 9
        else: # Secret was < 8, now in [6, 7]
            guess3 = 6
    # Path 2: Secret was < 5
    else:
        if guess2 < secret_number: # Secret was > 3, now is 4
            guess3 = 4
        else: # Secret was < 3, now in [1, 2]
            guess3 = 1

    if guess3 == secret_number:
        guess_log.append(f"猜 {guess3}: 恭喜猜對!")
        return True, guess_log
    else:
        if guess3 < secret_number:
            guess_log.append(f"猜 {guess3}: 錯，答案比{guess3}大")
        else:
            guess_log.append(f"猜 {guess3}: 錯，答案比{guess3}小")

    # If all 3 attempts fail
    return False, guess_log

# --- Main Simulation ---
total_runs = 100
wins = 0

for i in range(total_runs):
    secret = random.randint(1, 10)
    win, log = binary_search_player(secret)
    print(f"第{i+1}局，答案是 {secret}")
    for msg in log:
        print(msg)
    if win:
        print("結果：獲勝\n")
        wins += 1
    else:
        print("結果：失敗\n")

win_rate = (wins / total_runs) * 100
print(f"總勝率：{win_rate:.1f}%")
