#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <iostream>
#include "GL/freeglut.h"
#include "GL/gl.h"
#include "const.h"
#include "gpu.h"
//Starts up SDL and creates window
bool init();

//Loads media
bool loadMedia();

//Frees media and shuts down SDL
void close();

int size;
int offset;
//The window we'll be rendering to
SDL_Window *gWindow = NULL;

//The surface contained by the window
SDL_Surface *gScreenSurface = NULL;

//Current displayed PNG image
SDL_Surface *gPNGSurface = NULL;

std::string path;

void HandleFrame();

uint32_t * cudaMem;

void PreFrame()
{
	gPNGSurface = SDL_CreateRGBSurface( 0, SCREEN_WIDTH, SCREEN_HEIGHT, 32,
												0x00FF0000,
												0x0000FF00,
												0x000000FF,
												0xFF000000);	
	
	testRGB2HSL(0x00FF00C3);

	cudaMem = gpuAlloc(gPNGSurface->w,gPNGSurface->h);
	size = gPNGSurface->w * gPNGSurface->h * sizeof(int);
	gpuUp(gPNGSurface->pixels,cudaMem, size);
	offset = 0;	
	HandleFrame();
}

void HandleFrame()
{
	offset +=1;
	//std::cout << "The image is " << gScreenSurface->w << " by " << gScreenSurface->h <<"\n";

//	std::cout << "Made surface\n";
//	std::cout << sizeof(uint);
	checkError();
//	std::cout << "pushing data to device\n";
	checkError();
	gpuRender(cudaMem,gPNGSurface->w, gPNGSurface->h);
//	std::cout << gPNGSurface->h <<"\n";
	gpuBlit(cudaMem,gPNGSurface->pixels,size );
//	std::cout << gPNGSurface->h <<"\n";
	checkError();

	//Apply the PNG image
	SDL_BlitSurface(gPNGSurface, NULL, gScreenSurface, NULL);
	std::cout<<"a";
	//Update the surface
	SDL_UpdateWindowSurface(gWindow);
}

bool init()
{
	std::cout << "Beginning Program \n";
	//Initialization flag
	bool success = true;

	//Initialize SDL
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		printf("SDL could not initialize! SDL Error: %s\n", SDL_GetError());
		success = false;
	}
	else
	{
		//Create window
		gWindow = SDL_CreateWindow("SDL Tutorial", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
		if (gWindow == NULL)
		{
			printf("Window could not be created! SDL Error: %s\n", SDL_GetError());
			success = false;
		}
		else
		{
			//Initialize PNG loading
			int imgFlags = IMG_INIT_PNG;
			if (!(IMG_Init(imgFlags) & imgFlags))
			{
				printf("SDL_image could not initialize! SDL_image Error: %s\n", IMG_GetError());
				success = false;
			}
			else
			{
				//Get window surface
				gScreenSurface = SDL_GetWindowSurface(gWindow);
			}
		}
	}

	return success;
}

void close()
{
	//Free loaded image
	SDL_FreeSurface(gPNGSurface);
	gPNGSurface = NULL;

	//Destroy window
	SDL_DestroyWindow(gWindow);
	gWindow = NULL;

	//Quit SDL subsystems
	IMG_Quit();
	SDL_Quit();
}


void render(SDL_Surface *screen, void *cuda_pixels)
{
	gpuRender((uint32_t *)cuda_pixels, screen->w, screen->h);
	if (gpuBlit(cuda_pixels, screen->pixels, 0) != 0)
	{
		// todo: get cuda error
		std::cerr << "cuda error" << std::endl;
	};
}

int main(int argc, char *args[])
{
	std::cout << argc;
	path = args[1];
	std::cout << path;
	//Start up SDL and create window
	if (!init())
	{
		printf("Failed to initialize!\n");
	}
	else
	{
		//Load media
		if (!loadMedia())
		{
			printf("Failed to load media!\n");
		}
		else
		{
			//Main loop flag
			bool quit = false;

			//Event handler
			SDL_Event e;
			PreFrame();
			int counter = 0;
			//While application is running
			while (!quit)
			{
				//Handle events on queue
				while (SDL_PollEvent(&e) != 0)
				{
					//User requests quit
					if (e.type == SDL_QUIT)
					{
						quit = true;
					}
				//	HandleFrame();
				//	std::cout<<counter++<<'\n';
				}
			}
		}
	}

	//Free resources and close SDL
	close();

	return 0;
}