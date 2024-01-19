#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

using namespace std;
using namespace std::chrono;

int main()
{
    int numProcesses; // Currently unused, but could be implemented for parallelization
    string pattern;

    // Prompt for pattern input
    cout << "Enter the pattern to search (e.g., '%x%'): ";
    cin >> pattern;

    // Read the paragraph from the specified file
    ifstream file("words.txt");
    string paragraph;

    if (file.is_open())
    {
        getline(file, paragraph);
        file.close();
    }
    else
    {
        cerr << "Error opening file." << endl;
        return 1;
    }

    // Start time measurement
    auto start = high_resolution_clock::now();

    int count = 0;
    int i = 0;

    while (i < paragraph.length())
    {
        // Corrected pattern search
        i = paragraph.find_first_of(pattern, i);
        if (i == string::npos)
        {
            break;
        }
        // Ensure full pattern match
        if (paragraph.substr(i, pattern.length()) == pattern)
        {
            count++;
        }
        i += pattern.length();
    }

    // End time measurement
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    cout << "Total time taken: " << duration.count() << " microseconds" << endl;
    cout << "Number of occurrences of the pattern '" << pattern << "': " << count << endl;

    return 0;
}
