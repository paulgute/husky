#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <sstream>


bool sortbydes(const std::pair<std::string,int> &a, 
              const std::pair<std::string,int> &b) {
    return (a.second < b.second);
}

int main(int argc, char** argv) {
    // argv[1] base binary file name
    // argv[2] query binary file name
    // argv[3] data dimension
    // argv[4] top number
    int dimension = std::stoi(argv[3]);
    // store base data into vectors
	std::ifstream file1(argv[1]);
	std::string line1;
    int base_row_cnt = 0;
	std::vector<std::vector<std::string>> base_binary;
	while(std::getline(file1, line1)) {
		std::vector<std::string> base_row;
    	std::stringstream  linestream(line1);
    	std::string s;
    	std::getline(linestream, s); 
		std::string delimiter = ":";
		size_t pos = 0;
		std::string token;
		std::string ID;
		std::string binary;
		while ((pos = s.find(delimiter)) != std::string::npos) {
    		token = s.substr(0, pos);
			ID = token;
    		s.erase(0, pos + delimiter.length());
		}
		binary = s;
		base_row.push_back(ID);
		base_row.push_back(binary);
		base_binary.push_back(base_row);
		base_row.clear();
		base_row_cnt ++;
	}
    file1.close();

    // store query data into vectors
	std::ifstream file2(argv[2]);
	std::string line2;
    int query_row_cnt = 0;
	std::vector<std::vector<std::string>> query_binary;
	while(std::getline(file2, line2)) {
		std::vector<std::string> query_row;
    	std::stringstream  linestream(line2);
    	std::string s;
    	std::getline(linestream, s); 
        std::string delimiter = ":";
		size_t pos = 0;
		std::string token;
		std::string ID;
		std::string binary;
		while ((pos = s.find(delimiter)) != std::string::npos) {
    		token = s.substr(0, pos);
			ID = token;
    		s.erase(0, pos + delimiter.length());
		}
		binary = s;
		query_row.push_back(ID);
		query_row.push_back(binary);
		query_binary.push_back(query_row);
		query_row.clear();
		query_row_cnt ++;
	}
    file2.close();

    /*
	for (int i = 0; i < base_row_cnt; i++) {
		std::cout << base_binary[i][0] << ":" << base_binary[i][1] << std::endl;
	}
	for (int i = 0; i < query_row_cnt; i++) {
		std::cout << query_binary[i][0] << ":" << query_binary[i][1] << std::endl;
	}
    */
    int topk = std::stoi(argv[4]);
    // query the nearest
    for (int i = 0; i < query_row_cnt; i++) {
        std::vector<std::pair<std::string,int>> sort_list;
        std::string queryID = (query_binary[i])[0];
        for (int j = 0; j < base_row_cnt; j++) {
            int dist = 0;
            std::string ID = (base_binary[j])[0];
            std::string string1 = base_binary[j][1];
            std::string string2 = query_binary[i][1];
            for (int k = 0; k < dimension; k++) {
                if(string1[k] != string2[k]) {
                    dist += 1;
                }
            }
            sort_list.push_back(std::make_pair(ID, dist));
        }
        std::sort(sort_list.begin(), sort_list.end(), sortbydes);
        std::cout << queryID;
        for (int rank = 0; rank < topk; rank++) {
            std::cout << " " << (sort_list[rank]).first;
        }
        std::cout << std::endl;
    }

	return 0;
}
