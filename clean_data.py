def remove_quotation(line: str):
    new_line = line.replace('"', '')
    return new_line

# def remove_time(line: str):
#     return line.replace("T23:59:59.999Z", "")

def main():
    with open("bitcoin-price.csv") as file:
        with open("new.csv", "a") as output:
            for line in file:
                new_line = remove_quotation(line)

                # Keeping only the date in the timestamp column
                parts = new_line.rsplit("T23:59:59.999Z", 1)
                new_line = "".join(parts)

                output.write(new_line)
                # print(new_line)

if __name__ == "__main__":
    main()