#pragma once

int parseArguments(int argc, char ** argv, std::ofstream &file, bool &retflag);

void cannyTrial(std::ofstream &file, char ** argv, int i, int trial);
