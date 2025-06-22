
CXX = clang++
CXXFLAGS = -std=c++17 -I/opt/homebrew/include -I/opt/homebrew/Cellar/opencv/4.11.0_1/include/opencv4 -O3 -Wall -Wno-unused-but-set-variable -Wno-pessimizing-move
LDFLAGS = -L/opt/homebrew/lib -L/opt/homebrew/Cellar/opencv/4.11.0_1/lib \
           -ldlib -llapack -lblas \
           -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

SRC_DIR = .
SRCS = $(SRC_DIR)/main.cpp \
       $(SRC_DIR)/face_landmark_detector.cpp \
       $(SRC_DIR)/image_processor.cpp \
       $(SRC_DIR)/face_asymmetry_analyzer.cpp
OBJS = $(SRCS:.cpp=.o)


TARGET = face_landmark_detector
ASYMMETRY_TARGET = face_asymmetry_analysis


all: $(TARGET) $(ASYMMETRY_TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(ASYMMETRY_TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -DANALYZE_ASYMMETRY

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

format-check: 
	clang-format -n *.cpp *.h

format: 
	clang-format -i *.cpp *.h

clean:
	rm -f $(OBJS) $(TARGET) $(ASYMMETRY_TARGET) 
	rm -rf output output_csv

leaks: $(TARGET)
	leaks -atExit -- ./$(TARGET)
