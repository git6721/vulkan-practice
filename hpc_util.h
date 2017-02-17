#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>  //用于显示

#define GLM_FORCE_RADIANS  //？？
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // vulkan depth range 0-1.0  opengl -1.0-1.0
#include <glm/glm.hpp> //数学计算
#include <glm/gtc/matrix_transform.hpp>//控制模型的变换
#include <glm/gtx/hash.hpp>// 哈希

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>//load image  必须有上面的宏定义才能使用stb_image 该文件支持大多数 formats, like JPEG, PNG, BMP and GIF.

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>//load 3d model

//run debug
#ifdef NDEBUG   
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif