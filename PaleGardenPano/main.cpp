
extern int runFlowerKernel(int argc, char** argv);
extern int runFlowerKernelTextSeeds();
extern int runTreeKernel(int argc, char** argv);
extern int runTreeKernelTextSeeds();

int main(int argc, char** argv) {
	// runTreeKernelTextSeeds();
	runTreeKernel(argc, argv);

	return 0;
}