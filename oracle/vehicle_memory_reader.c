#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
	MAX_PATTERN_SIZE = 4096,
	SCAN_CHUNK_SIZE = 16 * 1024 * 1024,
};

static void fail(const char *message)
{
	fprintf(stderr, "vehicle memory reader: %s\n", message);
	exit(1);
}

static DWORD find_game_process(void)
{
	PROCESSENTRY32 entry;
	HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	DWORD result = 0;

	if (snapshot == INVALID_HANDLE_VALUE)
		fail("cannot enumerate processes");
	memset(&entry, 0, sizeof(entry));
	entry.dwSize = sizeof(entry);
	if (!Process32First(snapshot, &entry)) {
		CloseHandle(snapshot);
		fail("cannot read process list");
	}
	do {
		if (_stricmp(entry.szExeFile, "TmForever.exe") == 0) {
			if (result != 0) {
				CloseHandle(snapshot);
				fail("more than one TmForever.exe process is running");
			}
			result = entry.th32ProcessID;
		}
	} while (Process32Next(snapshot, &entry));
	CloseHandle(snapshot);
	if (result == 0)
		fail("TmForever.exe is not running");
	return result;
}

static uint8_t from_hex(char value)
{
	if (value >= '0' && value <= '9')
		return (uint8_t)(value - '0');
	if (value >= 'a' && value <= 'f')
		return (uint8_t)(value - 'a' + 10);
	if (value >= 'A' && value <= 'F')
		return (uint8_t)(value - 'A' + 10);
	fail("invalid hexadecimal digit");
	return 0;
}

static size_t decode_hex(
	const char *text, uint8_t *output, size_t output_capacity)
{
	size_t length = strlen(text);
	size_t size;

	while (length != 0 && isspace((unsigned char)text[length - 1]))
		--length;
	if ((length & 1) != 0)
		fail("hexadecimal input has odd length");
	size = length / 2;
	if (size == 0 || size > output_capacity)
		fail("hexadecimal input has invalid size");
	for (size_t i = 0; i < size; ++i)
		output[i] =
			(uint8_t)((from_hex(text[i * 2]) << 4) |
				from_hex(text[i * 2 + 1]));
	return size;
}

static int readable_protection(DWORD protect)
{
	DWORD base = protect & 0xff;
	if ((protect & PAGE_GUARD) != 0 || base == PAGE_NOACCESS)
		return 0;
	return base == PAGE_READONLY || base == PAGE_READWRITE
		|| base == PAGE_WRITECOPY || base == PAGE_EXECUTE_READ
		|| base == PAGE_EXECUTE_READWRITE
		|| base == PAGE_EXECUTE_WRITECOPY;
}

static void command_read(HANDLE process, const char *arguments)
{
	unsigned long address;
	unsigned long size;
	char trailing;
	uint8_t *bytes;
	SIZE_T copied = 0;

	if (sscanf(arguments, "%lx %lu %c", &address, &size, &trailing) != 2
	    || address > UINT32_MAX || size == 0 || size > UINT32_MAX)
		fail("invalid read command");
	bytes = (uint8_t *)malloc(size);
	if (bytes == NULL)
		fail("read allocation failed");
	if (!ReadProcessMemory(
		    process, (const void *)(uintptr_t)address, bytes, size,
		    &copied)
	    || copied != size) {
		free(bytes);
		fail("ReadProcessMemory failed");
	}
	printf("DATA ");
	for (size_t i = 0; i < size; ++i)
		printf("%02x", bytes[i]);
	printf("\n");
	fflush(stdout);
	free(bytes);
}

static void print_matches(
	const uint8_t *bytes, size_t size, uintptr_t address,
	const uint8_t *pattern, size_t pattern_size)
{
	if (size < pattern_size)
		return;
	for (size_t i = 0; i <= size - pattern_size; ++i) {
		if (bytes[i] == pattern[0]
		    && memcmp(bytes + i, pattern, pattern_size) == 0) {
			printf(" %08lx", (unsigned long)(address + i));
		}
	}
}

static void command_scan(HANDLE process, const char *arguments)
{
	uint8_t pattern[MAX_PATTERN_SIZE];
	size_t pattern_size =
		decode_hex(arguments, pattern, sizeof(pattern));
	uint8_t *buffer =
		(uint8_t *)malloc(SCAN_CHUNK_SIZE + pattern_size - 1);
	uintptr_t address = 0;
	MEMORY_BASIC_INFORMATION info;

	if (buffer == NULL)
		fail("scan allocation failed");
	printf("MATCHES");
	while (address <= UINT32_MAX
	       && VirtualQueryEx(
		       process, (const void *)address, &info,
		       sizeof(info)) == sizeof(info)) {
		uintptr_t base = (uintptr_t)info.BaseAddress;
		uintptr_t end = base + info.RegionSize;
		if (end <= base)
			break;
		if (info.State == MEM_COMMIT
		    && readable_protection(info.Protect)) {
			for (uintptr_t chunk = base; chunk < end;) {
				size_t request =
					(size_t)(end - chunk);
				SIZE_T copied = 0;
				if (request > SCAN_CHUNK_SIZE)
					request = SCAN_CHUNK_SIZE;
				if (ReadProcessMemory(
					    process, (const void *)chunk, buffer,
					    request, &copied)
				    && copied != 0) {
					print_matches(
						buffer, copied, chunk, pattern,
						pattern_size);
				}
				chunk += request;
			}
		}
		address = end;
	}
	printf("\n");
	fflush(stdout);
	free(buffer);
}

int main(void)
{
	char line[MAX_PATTERN_SIZE * 2 + 64];
	DWORD process_id = find_game_process();
	HANDLE process = OpenProcess(
		PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, process_id);

	if (process == NULL)
		fail("cannot open TmForever.exe");
	printf("READY %08lx\n", (unsigned long)process_id);
	fflush(stdout);
	while (fgets(line, sizeof(line), stdin) != NULL) {
		if (strncmp(line, "READ ", 5) == 0) {
			command_read(process, line + 5);
		} else if (strncmp(line, "SCAN ", 5) == 0) {
			command_scan(process, line + 5);
		} else if (strcmp(line, "QUIT\n") == 0) {
			CloseHandle(process);
			return 0;
		} else {
			fail("unknown command");
		}
	}
	CloseHandle(process);
	fail("command stream closed");
	return 1;
}
