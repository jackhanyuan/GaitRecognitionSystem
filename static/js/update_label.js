function updateLab(path, id) {
    fileName = path.split('\\');
    length = fileName.length;
    file = fileName[length - 1]
    document.getElementById(id).innerHTML = file;
};