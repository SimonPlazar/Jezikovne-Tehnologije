#include <iostream>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

using namespace std;

// Segments for pages
enum MainSegment {
    MAIN_FRONT,
    MAIN_BODY,
    MAIN_BACK
};

// Subsegments for paragraph
enum SubSegment {
    SUB_TITLE_PAGE,
    SUB_TOC,
    SUB_TOA,
    SUB_ABSTRACT_SLO,
    SUB_ABSTRACT_EN,
    SUB_ABSTRACT_DE,
    SUB_ABSTRACT,
    SUB_BIBLIOGRAPHY,
    SUB_ACRONYM,
    SUB_BODY,
    SUB_CONCLUSION,
    SUB_UNKNOWN
};

string subSegmentToString(SubSegment type) {
    switch (type) {
        case SUB_TITLE_PAGE:
            return "titlePage";
        case SUB_TOC:
            return "toc";
        case SUB_TOA:
            return "toa";
        case SUB_ABSTRACT_SLO:
            return "abstractSlo";
        case SUB_ABSTRACT_EN:
            return "abstractEn";
        case SUB_ABSTRACT_DE:
            return "abstractDe";
        case SUB_BIBLIOGRAPHY:
            return "bibliography";
        case SUB_ACRONYM:
            return "acronym";
        case SUB_BODY:
            return "body";
        case SUB_CONCLUSION:
            return "conclusion";
        default:
            return "unknown";
    }
}

string mainSegmentToString(MainSegment type) {
    switch (type) {
        case MAIN_FRONT:
            return "front";
        case MAIN_BODY:
            return "body";
        case MAIN_BACK:
            return "back";
        default:
            return "unknown";
    }
}

void scrapeFile(const string &input_filename, const string &output_filename) {
    // Open files with the determined filenames
    ifstream input(input_filename);
    ofstream output(output_filename);

    if (!input.is_open()) {
        cerr << "Error opening input file: " << input_filename << endl;
        return;
    }

    if (!output.is_open()) {
        cerr << "Error opening output file: " << output_filename << endl;
        return;
    }

    regex pageRegex("<page xml:id=\"(pb\\d+)\"([^>]*?)(/>|>)");
    regex pRegex("<p xml:id=\"(pb\\d+\\.p\\d+)\"[^>]*xml:lang=\"([^\"]*)\"[^>]*>([^<]+)</p>");
    regex endPageRegex("</page>");

    vector<string> paragraphLabelStrings;
    vector<string> pageLabelStrings;
    vector<string> chapterStrings;
    vector<string> bibliographyStrings;

    // State variables
    string currentPage;
    MainSegment currentMainSegment = MAIN_FRONT;
    SubSegment currentSubSegment = SUB_TITLE_PAGE;

    string line;
    while (getline(input, line)) {
        smatch match;

        // Check for page start
        if (regex_search(line, match, pageRegex)) {
            currentPage = match[1];
            bool isSelfClosing = (match[3] == "/>");

            // If self-closing
            if (isSelfClosing) {
                pageLabelStrings.push_back(currentPage + " " + mainSegmentToString(currentMainSegment));
                currentPage.clear();
            }
        }
        // Check for page end
        if (regex_search(line, match, endPageRegex)) {
            pageLabelStrings.push_back(currentPage + " " + mainSegmentToString(currentMainSegment));
            currentPage.clear();
            continue;
        }

        // Check for paragraph
        if (regex_search(line, match, pRegex)) {
            string pid = match[1];
            string lang = match[2];
            string content = match[3];

            if (currentSubSegment == SUB_TITLE_PAGE && currentPage != "pb1") {
                currentSubSegment = SUB_UNKNOWN;
                continue;
            }

            // check if chapter heading
            if (regex_search(content, regex("^\\s*\\d+(\\.\\d+)?\\s+[A-ZČŠŽĐ][^.]*$")) &&
                !regex_search(content, regex("\\.{3,}|\\d+$"))) {
                chapterStrings.push_back(currentPage + ": chapter " + content);

                currentSubSegment = SUB_UNKNOWN;

                regex uvodRegex("\\buvod\\b", regex::icase);
                if (regex_search(content, uvodRegex)) {
                    if (currentMainSegment != MAIN_BODY) {
                        currentMainSegment = MAIN_BODY;
                    }
                    cout << "Found uvod start on " << pid << endl;
                }

                regex virRegex("\\b(literatura|vir|viri|virov|seznam\\s+virov)\\b", regex::icase);
                if (regex_search(content, virRegex)) {
                    if (currentMainSegment != MAIN_BACK) {
                        currentMainSegment = MAIN_BACK;
                    }
                    currentSubSegment = SUB_BIBLIOGRAPHY;
                    cout << "Found vir start on " << pid << endl;
                }

                regex zakljucekRegex("\\bzaključek\\b", regex::icase);
                if (regex_search(content, zakljucekRegex)) {
                    currentSubSegment = SUB_CONCLUSION;
                    cout << "Found zakljucek start on " << pid << endl;
                }
            }

            regex kazaloRegex("\\bkazalo\\b", regex::icase);
            if (regex_search(content, kazaloRegex)) {
                currentSubSegment = SUB_TOC;
                cout << "Found kazalo start on " << pid << endl;
            }

            regex kraticeRegex("\\bkratice\\b", regex::icase);
            if (regex_search(content, kraticeRegex) &&
                (currentMainSegment == MAIN_FRONT || currentMainSegment == MAIN_BACK)) {
                currentSubSegment = SUB_ACRONYM;
                cout << "Found kratice start on " << pid << endl;
            }

            regex abstractRegex("\\b(povzetek|abstract|summary)\\b", regex::icase);
            if (regex_search(content, abstractRegex) &&
                !regex_search(content, regex("\\.{3,}|\\d+$"))) {
                currentSubSegment = SUB_ABSTRACT;
                cout << "Found abstract start on " << pid << endl;
            }

            regex keywordsRegex("\\b(KLJUČNE BESEDE|kljucne\\s+besede|ključne\\s+besede|key\\s+words)\\b",
                                regex::icase);
            regex odvecnaRegex("\\b(slik|slike|PREDGOVOR)\\b", regex::icase);
            if (regex_search(content, keywordsRegex) ||
                regex_search(content, odvecnaRegex)) {
                currentSubSegment = SUB_UNKNOWN;
            }

            if (currentSubSegment == SUB_ABSTRACT) {
                if (lang == "sl") {
                    paragraphLabelStrings.push_back(pid + " " + subSegmentToString(SUB_ABSTRACT_SLO));
                } else if (lang == "en") {
                    paragraphLabelStrings.push_back(pid + " " + subSegmentToString(SUB_ABSTRACT_EN));
                } else if (lang == "de") {
                    paragraphLabelStrings.push_back(pid + " " + subSegmentToString(SUB_ABSTRACT_DE));
                }
            } else if (currentSubSegment != SUB_UNKNOWN) {
                paragraphLabelStrings.push_back(pid + " " + subSegmentToString(currentSubSegment));

                if (currentSubSegment == SUB_BIBLIOGRAPHY) {
                    if (!regex_match(content, regex("^\\d+(\\.\\d+)?\\s+.*$"))) {
                        bibliographyStrings.push_back(content);
                    }
                }
            }
        }
    }

    // Output results
    output << "ID CLASS" << endl;
    for (const auto &entry: paragraphLabelStrings) {
        output << entry << endl;
    }
    output << endl;
    for (const auto &entry: pageLabelStrings) {
        output << entry << endl;
    }
    output << endl;
    for (const auto &entry: chapterStrings) {
        output << entry << endl;
    }
    output << endl;
    output << "bibliography:" << endl;
    for (const auto &entry: bibliographyStrings) {
        output << entry << endl;
    }

    input.close();
    output.close();
    cout << "Segmentation completed. Results in 'kas-4000.res'" << endl;
}

int main(int argc, char *argv[]) {
    string inputFilename;
    string outputFilename;

    // Default filenames
    string defaultInputFile = "../korpus/kas-4000.text.xml";
    string defaultOutputFile = "../kas-4000.res";

    // No command-line arguments
    cout << "Enter input filename (or press Enter for default '" << defaultInputFile << "'): ";
    getline(cin, inputFilename);

    if (inputFilename.empty()) {
        inputFilename = defaultInputFile;
    }

    cout << "Enter output filename (or press Enter for default '" << defaultOutputFile << "'): ";
    getline(cin, outputFilename);

    if (outputFilename.empty()) {
        outputFilename = defaultOutputFile;
    }


    scrapeFile(inputFilename, outputFilename);

    return 0;
}