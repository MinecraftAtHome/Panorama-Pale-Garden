
extern int runFlowerKernel(int argc, char** argv);
extern int runFlowerKernelTextSeeds();
extern int runTreeKernel(int argc, char** argv);
extern int runTreeKernelTextSeeds();

int main(int argc, char** argv) {
	// runFlowerKernel(argc, argv);
	// runTreeKernel(argc, argv);

	runTreeKernelTextSeeds();

	return 0;
}