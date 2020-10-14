#pragma once

#include <string>

std::string
default_log_dir()
{
    std::string cubemosLogDir = "";
#ifdef _WINDOWS
    cubemosLogDir = std::string(std::getenv("LOCALAPPDATA")) + "\\Cubemos\\SkeletonTracking\\logs";
#elif __linux__
    cubemosLogDir = std::string(std::getenv("HOME")) + "/.cubemos/skeleton_tracking/logs";
#endif
    return cubemosLogDir;
}

std::string
default_license_dir()
{
    std::string cubemosLicenseDir = "";
#ifdef _WINDOWS
    cubemosLicenseDir = std::string(std::getenv("LOCALAPPDATA")) + "\\Cubemos\\SkeletonTracking\\license";
#elif __linux__
    cubemosLicenseDir = std::string(std::getenv("HOME")) + "/.cubemos/skeleton_tracking/license";
#endif
    return cubemosLicenseDir;
}

std::string
default_model_dir()
{
    std::string cubemosModelDir = "";
#ifdef _WINDOWS
    cubemosModelDir = std::string(std::getenv("LOCALAPPDATA")) + "\\Cubemos\\SkeletonTracking\\models";
#elif __linux__
    cubemosModelDir = std::string(std::getenv("HOME")) + "/.cubemos/skeleton_tracking/models";
#endif
    return cubemosModelDir;
}

std::string
default_res_dir()
{
    std::string cubemosResDir = "";
#ifdef _WINDOWS
    cubemosResDir = std::string(std::getenv("LOCALAPPDATA")) + "\\Cubemos\\SkeletonTracking\\res";
#elif __linux__
    cubemosResDir = std::string(std::getenv("HOME")) + "/.cubemos/skeleton_tracking/res";
#endif
    return cubemosResDir;
}

#define CHECK_HANDLE_CREATION(retCode)                                                                                 \
    {                                                                                                                  \
        if (retCode == CM_FILE_DOES_NOT_EXIST) {                                                                       \
            std::cout << "Activation key does not exist. Please run the post installation script found"                \
                      << " in $CUBEMOS_SKEL_SDK / scripts to activate the license and "                                \
                      << " use it in your application." << std::endl;                                                  \
            CHECK_SUCCESS(retCode);                                                                                    \
        }                                                                                                              \
    }

#define CHECK_SUCCESS(retCode)                                                                                         \
    {                                                                                                                  \
        if (retCode != CM_SUCCESS) {                                                                                   \
            std::cerr << "Operation in file \"" << __FILE__ << "\" at line \"" << __LINE__                             \
                      << "\" failed with return code:  " << retCode << std::endl                                       \
                      << "Press any key to exit.." << std::endl;                                                       \
                                                                                                                       \
            std::cin.get();                                                                                            \
            return -1;                                                                                                 \
        }                                                                                                              \
    }

#define EXIT_PROGRAM(errMsg)                                                                                           \
    {                                                                                                                  \
        std::cerr << errMsg;                                                                                           \
        std::cin.get();                                                                                                \
        return -1;                                                                                                     \
    }
