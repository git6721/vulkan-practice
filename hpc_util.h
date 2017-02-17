#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>  //������ʾ

#define GLM_FORCE_RADIANS  //����
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // vulkan depth range 0-1.0  opengl -1.0-1.0
#include <glm/glm.hpp> //��ѧ����
#include <glm/gtc/matrix_transform.hpp>//����ģ�͵ı任
#include <glm/gtx/hash.hpp>// ��ϣ

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>//load image  ����������ĺ궨�����ʹ��stb_image ���ļ�֧�ִ���� formats, like JPEG, PNG, BMP and GIF.

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>//load 3d model

//run debug
#ifdef NDEBUG   
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif