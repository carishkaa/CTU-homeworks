idx=$1
# g++ -fsanitize=address -g -std=c++11 -O3 -Wall main.cpp -o main &&./main < pub"$idx".in && cat pub"$idx".out
g++ -g -std=c++11 -O3 -Wall zkouska.cpp -o main &&./main < datapub/pub"$idx".in && cat datapub/pub"$idx".out
