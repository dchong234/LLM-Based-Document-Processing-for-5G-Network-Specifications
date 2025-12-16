"""
Benchmark Questions for 5G Specification Testing
Defines test questions with ground truth answers for evaluating fine-tuned model.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Benchmark questions covering various 5G topics
BENCHMARK_QUESTIONS = [
    # ========================================================================
    # EASY QUESTIONS (5) - Basic definitions and terminology
    # ========================================================================
    {
        "id": "E001",
        "question": "What is 5G NR?",
        "ground_truth": "5G NR (New Radio) is the radio access technology standard for 5G networks. It defines the physical layer and medium access control layer specifications for 5G wireless communication systems.",
        "category": "Terminology",
        "difficulty": "Easy",
        "keywords": ["5G NR", "New Radio", "radio access technology", "physical layer", "MAC layer"]
    },
    {
        "id": "E002",
        "question": "What are the two main frequency ranges in 5G?",
        "ground_truth": "5G operates in two main frequency ranges: FR1 (Frequency Range 1) covering sub-6 GHz bands (410 MHz to 7.125 GHz) and FR2 (Frequency Range 2) covering millimeter wave bands (24.25 GHz to 52.6 GHz).",
        "category": "Frequency Bands",
        "difficulty": "Easy",
        "keywords": ["FR1", "FR2", "frequency ranges", "sub-6 GHz", "millimeter wave", "mmWave"]
    },
    {
        "id": "E003",
        "question": "What is numerology in 5G?",
        "ground_truth": "Numerology in 5G refers to the subcarrier spacing configuration. It defines the spacing between subcarriers in the OFDM waveform, with values μ (mu) ranging from 0 to 4, corresponding to subcarrier spacings of 15 kHz, 30 kHz, 60 kHz, 120 kHz, and 240 kHz respectively.",
        "category": "Physical Layer",
        "difficulty": "Easy",
        "keywords": ["numerology", "subcarrier spacing", "OFDM", "mu", "μ"]
    },
    {
        "id": "E004",
        "question": "What is the difference between 5G NSA and SA?",
        "ground_truth": "5G NSA (Non-Standalone) uses the existing 4G LTE core network (EPC) for control plane functions while using 5G NR for the radio access. 5G SA (Standalone) uses a new 5G core network (5GC) for both control and user plane functions with 5G NR radio access.",
        "category": "Architecture",
        "difficulty": "Easy",
        "keywords": ["NSA", "SA", "Non-Standalone", "Standalone", "5GC", "EPC", "core network"]
    },
    {
        "id": "E005",
        "question": "What is a resource block in 5G?",
        "ground_truth": "A resource block (RB) in 5G is the basic unit of resource allocation in the frequency domain. It consists of 12 consecutive subcarriers in the frequency domain. The size of a resource block depends on the numerology used.",
        "category": "Resource Allocation",
        "difficulty": "Easy",
        "keywords": ["resource block", "RB", "subcarriers", "frequency domain", "resource allocation"]
    },
    
    # ========================================================================
    # MEDIUM QUESTIONS (8) - Technical details and concepts
    # ========================================================================
    {
        "id": "M001",
        "question": "What modulation schemes are used in 5G NR?",
        "ground_truth": "5G NR uses QPSK (Quadrature Phase Shift Keying) for control channels and low data rates, 16-QAM (Quadrature Amplitude Modulation) for medium data rates, 64-QAM for high data rates, and 256-QAM for very high data rates in good channel conditions. π/2-BPSK is also used for some control channels.",
        "category": "Modulation",
        "difficulty": "Medium",
        "keywords": ["modulation", "QPSK", "QAM", "16-QAM", "64-QAM", "256-QAM", "BPSK", "modulation schemes"]
    },
    {
        "id": "M002",
        "question": "Explain the concept of bandwidth parts in 5G NR.",
        "ground_truth": "A bandwidth part (BWP) in 5G NR is a contiguous set of physical resource blocks configured within a carrier. A UE can be configured with multiple BWPs, but only one can be active at a time. BWPs allow for flexible bandwidth allocation, power saving, and support for different numerologies within the same carrier.",
        "category": "Bandwidth",
        "difficulty": "Medium",
        "keywords": ["bandwidth part", "BWP", "resource blocks", "carrier", "numerology", "UE"]
    },
    {
        "id": "M003",
        "question": "What is HARQ in 5G and how does it work?",
        "ground_truth": "HARQ (Hybrid Automatic Repeat Request) is an error correction mechanism in 5G that combines forward error correction (FEC) with automatic repeat request (ARQ). When a packet is received with errors, the receiver stores it and requests a retransmission. The retransmitted data is combined with the previously received data using soft combining to improve the probability of successful decoding.",
        "category": "Protocols",
        "difficulty": "Medium",
        "keywords": ["HARQ", "Hybrid ARQ", "error correction", "FEC", "ARQ", "retransmission", "soft combining"]
    },
    {
        "id": "M004",
        "question": "What are the different types of reference signals in 5G NR?",
        "ground_truth": "5G NR uses several types of reference signals: DMRS (Demodulation Reference Signal) for channel estimation during data demodulation, CSI-RS (Channel State Information Reference Signal) for channel quality measurement and feedback, SRS (Sounding Reference Signal) for uplink channel quality estimation, and PT-RS (Phase Tracking Reference Signal) for phase tracking in high-frequency bands.",
        "category": "Reference Signals",
        "difficulty": "Medium",
        "keywords": ["reference signals", "DMRS", "CSI-RS", "SRS", "PT-RS", "channel estimation", "channel state information"]
    },
    {
        "id": "M005",
        "question": "Explain beamforming in 5G NR.",
        "ground_truth": "Beamforming in 5G NR is a technique that uses multiple antennas to focus radio frequency energy in specific directions, creating directional beams instead of broadcasting in all directions. This improves signal strength, reduces interference, and increases capacity. 5G NR supports both analog and digital beamforming, with beam management procedures for beam selection, refinement, and recovery.",
        "category": "Beamforming",
        "difficulty": "Medium",
        "keywords": ["beamforming", "beams", "antenna", "directional", "beam management", "analog beamforming", "digital beamforming"]
    },
    {
        "id": "M006",
        "question": "What is the 5G network architecture and its main components?",
        "ground_truth": "The 5G network architecture consists of three main components: the User Equipment (UE), the Radio Access Network (RAN) with gNodeB base stations, and the 5G Core (5GC). The 5GC includes network functions like AMF (Access and Mobility Management Function), SMF (Session Management Function), UPF (User Plane Function), PCF (Policy Control Function), and AUSF (Authentication Server Function).",
        "category": "Architecture",
        "difficulty": "Medium",
        "keywords": ["architecture", "5GC", "RAN", "gNodeB", "AMF", "SMF", "UPF", "PCF", "AUSF", "network functions"]
    },
    {
        "id": "M007",
        "question": "What is network slicing in 5G?",
        "ground_truth": "Network slicing in 5G allows the creation of multiple virtual networks on top of a shared physical infrastructure. Each slice can have different characteristics (latency, bandwidth, reliability) tailored to specific use cases like enhanced mobile broadband (eMBB), ultra-reliable low-latency communications (URLLC), or massive machine-type communications (mMTC).",
        "category": "Network Slicing",
        "difficulty": "Medium",
        "keywords": ["network slicing", "slices", "virtual networks", "eMBB", "URLLC", "mMTC", "use cases"]
    },
    {
        "id": "M008",
        "question": "What are the different channel types in 5G NR?",
        "ground_truth": "5G NR has three types of channels: logical channels (define what type of data is transmitted), transport channels (define how data is transmitted), and physical channels (define where data is transmitted over the air). Key physical channels include PDCCH (Physical Downlink Control Channel), PDSCH (Physical Downlink Shared Channel), PUCCH (Physical Uplink Control Channel), and PUSCH (Physical Uplink Shared Channel).",
        "category": "Channels",
        "difficulty": "Medium",
        "keywords": ["channels", "logical channels", "transport channels", "physical channels", "PDCCH", "PDSCH", "PUCCH", "PUSCH"]
    },
    
    # ========================================================================
    # HARD QUESTIONS (7) - Complex interactions and procedures
    # ========================================================================
    {
        "id": "H001",
        "question": "Explain the complete HARQ process including timing and feedback mechanisms in 5G NR.",
        "ground_truth": "The HARQ process in 5G NR involves: (1) The transmitter sends a transport block with a redundancy version (RV), (2) The receiver attempts decoding and sends ACK/NACK feedback, (3) If NACK, the transmitter sends a retransmission with a different RV, (4) The receiver combines the original and retransmitted data using soft combining, (5) The process repeats until successful or maximum retransmissions. Timing is controlled by K1 (feedback timing) and K2 (retransmission timing) offsets, with asynchronous HARQ in downlink and synchronous HARQ in uplink.",
        "category": "Protocols",
        "difficulty": "Hard",
        "keywords": ["HARQ", "process", "timing", "feedback", "ACK", "NACK", "redundancy version", "RV", "soft combining", "K1", "K2", "asynchronous", "synchronous"]
    },
    {
        "id": "H002",
        "question": "How does beam management work in 5G NR, including initial beam acquisition, beam refinement, and beam recovery?",
        "ground_truth": "Beam management in 5G NR involves three main procedures: (1) Initial beam acquisition uses SSB (Synchronization Signal Block) beams transmitted in different directions, with the UE measuring and reporting the best beam, (2) Beam refinement uses CSI-RS for more precise beam tracking and adjustment, with periodic or aperiodic beam measurements, (3) Beam recovery is triggered when beam quality degrades below a threshold, involving beam failure detection, candidate beam identification, and beam failure recovery request procedures to re-establish communication.",
        "category": "Beamforming",
        "difficulty": "Hard",
        "keywords": ["beam management", "beam acquisition", "beam refinement", "beam recovery", "SSB", "CSI-RS", "beam failure detection", "beam failure recovery"]
    },
    {
        "id": "H003",
        "question": "Explain the complete random access procedure in 5G NR, including contention-based and contention-free access.",
        "ground_truth": "The random access procedure in 5G NR has two types: (1) Contention-based random access (CBRA) involves four steps: (a) UE sends PRACH preamble (Msg1), (b) gNB responds with Random Access Response (Msg2) containing timing advance and temporary C-RNTI, (c) UE sends RRC connection request (Msg3) using the allocated resources, (d) gNB sends contention resolution (Msg4). (2) Contention-free random access (CFRA) is used for handover and uses a dedicated preamble assigned by the network, skipping contention resolution. The procedure supports both 2-step and 4-step RACH.",
        "category": "Procedures",
        "difficulty": "Hard",
        "keywords": ["random access", "RACH", "PRACH", "contention-based", "contention-free", "CBRA", "CFRA", "Msg1", "Msg2", "Msg3", "Msg4", "preamble", "2-step RACH", "4-step RACH"]
    },
    {
        "id": "H004",
        "question": "How does 5G NR handle multi-carrier aggregation and what are the different aggregation scenarios?",
        "ground_truth": "5G NR supports carrier aggregation (CA) where multiple component carriers (CCs) are aggregated to increase bandwidth. Scenarios include: (1) Intra-band contiguous CA (adjacent carriers in same band), (2) Intra-band non-contiguous CA (non-adjacent carriers in same band), (3) Inter-band CA (carriers in different bands). The primary cell (PCell) provides control signaling, while secondary cells (SCells) provide additional capacity. CA supports up to 16 component carriers with cross-carrier scheduling and independent numerology per carrier.",
        "category": "Carrier Aggregation",
        "difficulty": "Hard",
        "keywords": ["carrier aggregation", "CA", "component carriers", "CC", "PCell", "SCell", "intra-band", "inter-band", "contiguous", "non-contiguous", "cross-carrier scheduling"]
    },
    {
        "id": "H005",
        "question": "Explain the complete scheduling and resource allocation process in 5G NR, including DCI formats and scheduling types.",
        "ground_truth": "Scheduling in 5G NR works as follows: (1) The gNB sends DCI (Downlink Control Information) on PDCCH indicating resource allocation, (2) DCI formats include DCI 1_0/1_1 for downlink scheduling and DCI 0_0/0_1 for uplink scheduling, (3) Resource allocation uses frequency domain resource assignment (FDRA) and time domain resource assignment (TDRA), (4) Scheduling types include dynamic scheduling (per-TTI allocation), semi-persistent scheduling (periodic allocation), and configured grant (pre-configured resources), (5) The UE monitors PDCCH in search spaces and decodes DCI to determine allocated resources for PDSCH or PUSCH transmission.",
        "category": "Scheduling",
        "difficulty": "Hard",
        "keywords": ["scheduling", "resource allocation", "DCI", "PDCCH", "PDSCH", "PUSCH", "FDRA", "TDRA", "dynamic scheduling", "semi-persistent scheduling", "SPS", "configured grant", "search space"]
    },
    {
        "id": "H006",
        "question": "How does 5G NR implement dual connectivity and what are the different deployment scenarios?",
        "ground_truth": "Dual connectivity (DC) in 5G NR allows a UE to simultaneously connect to two base stations: a master node (MN) and a secondary node (SN). Deployment scenarios include: (1) EN-DC (E-UTRA-NR Dual Connectivity) with LTE eNB as MN and 5G gNB as SN, (2) NE-DC (NR-E-UTRA Dual Connectivity) with 5G gNB as MN and LTE eNB as SN, (3) NGEN-DC (NG-RAN E-UTRA-NR Dual Connectivity) with both nodes connected to 5GC, (4) NR-DC (NR-NR Dual Connectivity) with two 5G gNBs. The MN handles control plane, while data can be split between MN and SN for increased throughput.",
        "category": "Dual Connectivity",
        "difficulty": "Hard",
        "keywords": ["dual connectivity", "DC", "master node", "MN", "secondary node", "SN", "EN-DC", "NE-DC", "NGEN-DC", "NR-DC", "control plane", "data split"]
    },
    {
        "id": "H007",
        "question": "Explain the complete handover procedure in 5G NR, including measurement configuration, handover decision, and execution phases.",
        "ground_truth": "The handover procedure in 5G NR involves: (1) Measurement configuration phase where the source gNB configures the UE with measurement objects (frequencies, cells), reporting configurations, and measurement gaps, (2) Measurement reporting phase where the UE performs measurements (RSRP, RSRQ, SINR) and sends measurement reports when triggers are met, (3) Handover decision phase where the source gNB evaluates reports and decides to initiate handover, (4) Handover preparation where the source gNB requests handover from target gNB via Xn interface, (5) Handover execution where the source gNB sends RRC reconfiguration with mobility control info, UE performs random access to target, and target confirms completion, (6) Handover completion where data forwarding occurs and the source releases resources.",
        "category": "Procedures",
        "difficulty": "Hard",
        "keywords": ["handover", "measurement", "RSRP", "RSRQ", "SINR", "measurement gaps", "RRC reconfiguration", "mobility control", "Xn interface", "data forwarding"]
    },
]


def get_questions_by_difficulty(difficulty: str = None) -> List[Dict[str, Any]]:
    """
    Get questions filtered by difficulty.
    
    Args:
        difficulty: Difficulty level to filter by ("Easy", "Medium", "Hard", or None for all)
    
    Returns:
        List of question dictionaries
    """
    if difficulty is None:
        return BENCHMARK_QUESTIONS
    
    return [q for q in BENCHMARK_QUESTIONS if q['difficulty'] == difficulty]


def get_questions_by_category(category: str = None) -> List[Dict[str, Any]]:
    """
    Get questions filtered by category.
    
    Args:
        category: Category to filter by (or None for all)
    
    Returns:
        List of question dictionaries
    """
    if category is None:
        return BENCHMARK_QUESTIONS
    
    return [q for q in BENCHMARK_QUESTIONS if q['category'] == category]


def get_question_by_id(question_id: str) -> Dict[str, Any]:
    """
    Get a specific question by ID.
    
    Args:
        question_id: Question ID (e.g., "E001", "M005", "H003")
    
    Returns:
        Question dictionary or None if not found
    """
    for question in BENCHMARK_QUESTIONS:
        if question['id'] == question_id:
            return question
    return None


def save_questions_to_json(output_path: Path):
    """
    Save benchmark questions to JSON file.
    
    Args:
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(BENCHMARK_QUESTIONS, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Benchmark questions saved to: {output_path}")


def load_questions_from_json(input_path: Path) -> List[Dict[str, Any]]:
    """
    Load benchmark questions from JSON file.
    
    Args:
        input_path: Path to JSON file
    
    Returns:
        List of question dictionaries
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    return questions


def print_question_statistics():
    """Print statistics about benchmark questions."""
    total = len(BENCHMARK_QUESTIONS)
    easy = len(get_questions_by_difficulty("Easy"))
    medium = len(get_questions_by_difficulty("Medium"))
    hard = len(get_questions_by_difficulty("Hard"))
    
    categories = {}
    for q in BENCHMARK_QUESTIONS:
        cat = q['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n{'=' * 60}")
    print("Benchmark Questions Statistics")
    print(f"{'=' * 60}")
    print(f"Total questions: {total}")
    print(f"\nBy Difficulty:")
    print(f"  Easy: {easy}")
    print(f"  Medium: {medium}")
    print(f"  Hard: {hard}")
    print(f"\nBy Category:")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")
    print(f"{'=' * 60}\n")


def print_questions_summary():
    """Print summary of all questions."""
    print(f"\n{'=' * 60}")
    print("Benchmark Questions Summary")
    print(f"{'=' * 60}\n")
    
    for question in BENCHMARK_QUESTIONS:
        print(f"[{question['id']}] ({question['difficulty']}) - {question['category']}")
        print(f"  Q: {question['question']}")
        print(f"  Keywords: {', '.join(question['keywords'])}")
        print()


def main():
    """Main function to display and manage benchmark questions."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Manage benchmark questions for 5G specification testing"
    )
    parser.add_argument(
        '--difficulty',
        type=str,
        choices=['Easy', 'Medium', 'Hard'],
        default=None,
        help='Filter questions by difficulty'
    )
    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='Filter questions by category'
    )
    parser.add_argument(
        '--id',
        type=str,
        default=None,
        help='Get specific question by ID'
    )
    parser.add_argument(
        '--save-json',
        type=str,
        default=None,
        help='Save questions to JSON file'
    )
    parser.add_argument(
        '--statistics',
        action='store_true',
        help='Print statistics about questions'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary of all questions'
    )
    
    args = parser.parse_args()
    
    # Save to JSON if requested
    if args.save_json:
        save_questions_to_json(Path(args.save_json))
        return 0
    
    # Print statistics
    if args.statistics:
        print_question_statistics()
        return 0
    
    # Print summary
    if args.summary:
        print_questions_summary()
        return 0
    
    # Get specific question by ID
    if args.id:
        question = get_question_by_id(args.id)
        if question:
            print(f"\n{'=' * 60}")
            print(f"Question {question['id']}")
            print(f"{'=' * 60}")
            print(f"Difficulty: {question['difficulty']}")
            print(f"Category: {question['category']}")
            print(f"\nQuestion: {question['question']}")
            print(f"\nGround Truth Answer:")
            print(question['ground_truth'])
            print(f"\nKeywords: {', '.join(question['keywords'])}")
            print(f"{'=' * 60}\n")
        else:
            print(f"Question with ID '{args.id}' not found.")
        return 0
    
    # Filter and display questions
    questions = BENCHMARK_QUESTIONS
    
    if args.difficulty:
        questions = get_questions_by_difficulty(args.difficulty)
    
    if args.category:
        questions = get_questions_by_category(args.category)
    
    # Print filtered questions
    print(f"\n{'=' * 60}")
    print("Benchmark Questions")
    print(f"{'=' * 60}")
    if args.difficulty:
        print(f"Difficulty: {args.difficulty}")
    if args.category:
        print(f"Category: {args.category}")
    print(f"Total: {len(questions)} questions")
    print(f"{'=' * 60}\n")
    
    for question in questions:
        print(f"[{question['id']}] ({question['difficulty']}) - {question['category']}")
        print(f"  {question['question']}")
        print()
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

