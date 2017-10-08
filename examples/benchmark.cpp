#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <iterator>
#include <algorithm>

bool sortbydes(const std::pair<int,float> &a,
              const std::pair<int,float> &b) {
    return (a.second < b.second);
}

int main(int argc, char** argv) {
    // argv[1] base file name
    // argv[2] query file name
    // argv[3] data dimension
    // argv[4] top number
    int dimension = std::stoi(argv[3]);
    // store base data into vectors
	float f;
    std::ifstream file1(argv[1], std::ios::binary);
    int cur_col = 0;
    int base_row_cnt = 0;
    std::vector<std::vector<float>> base_dataset;
    std::vector<float> base_row(dimension + 1);
    while (file1.read(reinterpret_cast<char*>(&f), sizeof(float))) {
        if (cur_col != 0) {
            base_row[cur_col] = f;
        }
        else {
            base_row[cur_col] = base_row_cnt;
        }
        cur_col += 1;
        if( cur_col == (dimension + 1)) {
            cur_col = 0;
            base_dataset.push_back(base_row);
            base_row_cnt += 1;
        }
	}
    file1.close();

    std::ifstream file2(argv[2], std::ios::binary);
    cur_col = 0;
    int query_row_cnt = 0;
    std::vector<std::vector<float>> query_dataset;
    std::vector<float> query_row(dimension + 1);
    while (file2.read(reinterpret_cast<char*>(&f), sizeof(float))) {
        if (cur_col != 0) {
            query_row[cur_col] = f;
        }
        else {
            query_row[cur_col] = query_row_cnt;
        }
        cur_col += 1;
        if( cur_col == (dimension + 1)) {
            cur_col = 0;
            query_dataset.push_back(query_row);
            query_row_cnt += 1;
        }
	}
    file2.close();
    
    int topk = std::stoi(argv[4]);
    // query the nearest
    for (int i = 0; i < query_row_cnt; i++) {
        std::vector<std::pair<int,float>> sort_list;
		int queryID = (query_dataset[i])[0];
        for (int j = 0; j < base_row_cnt; j++) {
            float dist = 0;
            int ID = (base_dataset[j])[0];
            for (int k = 1; k < dimension + 1; k++) {
                float diff = ((query_dataset[i])[k] - (base_dataset[j])[k]);
                dist += pow(diff,2);
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
