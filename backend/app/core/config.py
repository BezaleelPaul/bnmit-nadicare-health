class Settings:
    PROJECT_NAME: str = "SOMYTHRA Core"
    PROJECT_IDENTITY: str = "SOMYTHRA"
    VERSION: str = "SOMYTHRA-181126-001"
    DESCRIPTION: str = (
        "Physiological intelligence & digital-twin research engine. "
        "The science of understanding the human body as an interconnected living system."
    )
    API_V1_STR: str = "/api/v1"
    
    # Tanaka Max HR formula: 208 - (0.7 * age)
    TANAKA_BASE: float = 208.0
    TANAKA_FACTOR: float = 0.7
    
    # Safety thresholds
    CRITICAL_HR_RATIO: float = 0.90
    WARNING_HR_RATIO: float = 0.75
    CRITICAL_HRV_MS: float = 25.0
    WARNING_HRV_MS: float = 35.0
    
    # Valid measurement bounds for sanitization
    MIN_HR: float = 20.0
    MAX_HR: float = 250.0
    MIN_HRV: float = 0.0
    MAX_HRV: float = 250.0

settings = Settings()
